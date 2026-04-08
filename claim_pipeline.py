"""AI Insurance Claim Decision System — main entry point.

Reads a PDF insurance document, sends it to an LLM via the OpenAI-compatible
Hugging Face Inference API, and evaluates the decision through an OpenEnv
reward function.

Usage:
    python inference.py                 # uses sample.pdf by default
    python inference.py path/to/doc.pdf # use a custom PDF
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from pdf_utils import extract_text_from_pdf
from env import InsuranceEnv
from insurer_policy_loader import (
    detect_insurer_from_text,
    load_policy_criteria,
    format_criteria_for_prompt,
)
from fraud_detector import validate_document_consistency
from claim_assessor import (
    extract_structured_claim_data,
    evaluate_policy_fit,
    parse_llm_decision,
    load_claims_history,
    save_claims_history,
    history_consistency_score,
    compute_confidence_score,
    final_recommendation,
)
from feedback_handler import (
    adjust_confidence_for_feedback,
    apply_feedback_to_decision,
    build_claim_snapshot,
    load_feedback_log,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # load from .env if present

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
_client = None


def get_llm_client() -> OpenAI:
    """Lazily create OpenAI-compatible client for Hugging Face Inference."""
    global _client
    if _client is not None:
        return _client

    if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
        raise RuntimeError(
            "Missing environment variables. Set API_BASE_URL, MODEL_NAME, and "
            "HF_TOKEN (or create a .env file)."
        )

    _client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )
    return _client

# OpenEnv environment
env = InsuranceEnv()

# Memory file — stores past decisions so the LLM can learn from them
MEMORY_FILE = Path(__file__).parent / "memory.json"


# ---------------------------------------------------------------------------
# Memory / Learning Effect
# ---------------------------------------------------------------------------
def load_memory() -> list:
    """Load past decisions from disk."""
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            raw_history = json.load(f)
        return normalize_history(raw_history)
    return []


def save_memory(history: list) -> None:
    """Persist decision history to disk (keep latest 20)."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-20:], f, indent=2)


def normalize_history(history: list) -> list:
    """Recompute rewards for existing memory entries to fix legacy bad scores."""
    normalized = []
    for item in history:
        answer = (item or {}).get("answer", "")
        if not answer:
            continue
        _, recomputed_reward, _, _ = env.step(answer)
        normalized.append({"answer": answer, "reward": recomputed_reward})
    return normalized


def best_examples(history: list, top_n: int = 3) -> str:
    """Return the highest-reward past examples as few-shot context."""
    if not history:
        return ""
    ranked = sorted(history, key=lambda h: h["reward"], reverse=True)[:top_n]
    lines = [
        "\nBelow are examples of previous high-quality evaluations for "
        "reference only. Do NOT copy them — evaluate the NEW document below."
    ]
    for i, ex in enumerate(ranked, 1):
        lines.append(f"\n--- Example {i} (reward {ex['reward']}) ---\n{ex['answer']}")
    lines.append("\n--- End of examples ---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------
def generate_prompt(
    document_text: str, few_shot: str = "", policy_criteria: dict = None
) -> str:
    """Build the evaluation prompt for the LLM with optional insurer policy criteria."""
    criteria_section = ""
    if policy_criteria:
        criteria_section = format_criteria_for_prompt(policy_criteria)
    
    return f"""You are an insurance claim evaluator.

Rules:
- Approve if the policy clearly covers the treatment or procedure.
- Reject if the policy does not cover it or there is a mismatch.
- Apply insurer-specific policy criteria (see below) when making decisions.

Return strictly in this format:
Decision: Approved or Rejected
Reason: <short explanation citing policy coverage/exclusions>

{criteria_section}

{few_shot}
NOW evaluate the following NEW document:

Document:
{document_text}
"""


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def _mock_llm_answer(policy_fit: dict, extracted_data: dict) -> str:
    """Deterministic fallback decision path used for reproducible baseline runs."""
    fit_score = policy_fit.get("score", 0.0)
    diagnosis = extracted_data.get("diagnosis") or "the reported diagnosis"
    treatment = extracted_data.get("treatment") or extracted_data.get("procedure") or "the treatment"

    if fit_score >= 0.7:
        return (
            "Decision: Approved\n"
            "Reason: Policy criteria checks are mostly satisfied, claim amount is within "
            "sum insured limits, and the treatment appears covered for "
            f"{diagnosis} using {treatment}."
        )

    return (
        "Decision: Rejected\n"
        "Reason: Policy fit checks indicate missing or failed requirements "
        "for this claim, so approval conditions are not met."
    )


def run_inference(pdf_path: str, use_mock_model: bool = False) -> dict:
    """End-to-end inference pipeline with fraud detection & insurer-aware matching.

    1. Extract text from the PDF.
    2. Validate document for fraud indicators.
    3. Detect insurer from document.
    4. Load matching policy criteria.
    5. Reset the OpenEnv environment.
    6. Load memory and build prompt with criteria + few-shot examples.
    7. Call the LLM.
    8. Score the decision with the environment reward function.
    9. Save the result to memory for future learning.
    10. Return structured result with insurer & fraud info.
    """
    # Step 1 — PDF extraction
    document_text = extract_text_from_pdf(pdf_path)
    if not document_text:
        return {"error": "No text could be extracted from the PDF."}

    # Step 2 — Extract structured JSON data from document text
    extracted_data = extract_structured_claim_data(document_text)

    # Step 3 — Validate document for fraud indicators
    is_valid, fraud_flags = validate_document_consistency(document_text)
    fraud_detected = len(fraud_flags) > 0
    claims_history = load_claims_history()
    feedback_log = load_feedback_log()
    precheck_snapshot = build_claim_snapshot(
        {
            "insurer_detected": extracted_data.get("insurer", "Unknown"),
            "fraud_detected": fraud_detected,
            "fraud_reasons": fraud_flags,
            "structured_claim_data": extracted_data,
        }
    )
    
    if fraud_detected:
        # Allow strong prior human feedback to override repeated auto-rejections.
        feedback_decision = apply_feedback_to_decision(
            predicted_decision="Rejected",
            predicted_confidence=0.99,
            claim_snapshot=precheck_snapshot,
            feedback_log=feedback_log,
        )
        feedback_applied = feedback_decision.get("applied", False)
        final_decision = feedback_decision.get("decision", "Rejected")
        final_confidence = feedback_decision.get("confidence", 0.99)
        fraud_reason = "\n".join([f"• {flag}" for flag in fraud_flags])

        if feedback_applied:
            reason = (
                "Similar prior human feedback changed this outcome. "
                f"Original rule-based validation rejected the file for:\n{fraud_reason}"
            )
            decision_output = (
                f"Decision: {final_decision}\n"
                f"Reason: {reason}"
            )
            final_fraud_detected = False
            recommendation = final_recommendation(final_decision, final_confidence, False, reason)
            reward = 0.7 if final_decision == "Approved" else 0.6
        else:
            decision_output = (
                "Decision: Rejected\n"
                "Reason: Document validation failed - suspected fraudulent claim:\n"
                f"{fraud_reason}"
            )
            final_fraud_detected = True
            recommendation = final_recommendation("Rejected", 0.99, True, f"Document validation failed - suspected fraudulent claim: {fraud_reason}")
            reward = 0.0

        result = {
            "insurer_detected": extracted_data.get("insurer") or "Unknown (Fraud Detected)",
            "policy_criteria_applied": False,
            "fraud_detected": final_fraud_detected,
            "rule_engine_fraud_detected": True,
            "fraud_reasons": fraud_flags,
            "decision_output": decision_output,
            "reward": reward,
            "memory_size": 0,
            "parsed_decision": final_decision,
            "parsed_reason": decision_output.split("Reason:", 1)[-1].strip(),
            "model_decision": "Rejected",
            "claim_will_be_approved": final_decision == "Approved" and final_confidence >= 0.7,
            "is_document_correct": not final_fraud_detected,
            "confidence_score": final_confidence,
            "final_recommendation": recommendation,
            "feedback_override_applied": feedback_applied,
            "feedback_override_summary": feedback_decision.get("summary", ""),
            "structured_claim_data": extracted_data,
            "policy_fit": {
                "score": 0.0,
                "checks": [],
                "passed_checks": 0,
                "total_checks": 0,
            },
            "history_consistency": 0.0,
        }
        claims_history.append(result)
        save_claims_history(claims_history)
        return result

    # Step 4 — Detect insurer from document
    detected_insurer = detect_insurer_from_text(document_text)

    # Step 5 — Load policy criteria for the detected insurer
    policy_criteria = {}
    if detected_insurer:
        policy_criteria = load_policy_criteria(detected_insurer)

    # Step 6 — Evaluate extracted JSON against policy criteria
    policy_fit = evaluate_policy_fit(extracted_data, policy_criteria)

    # Step 7 — Environment reset
    state = env.reset(document_text)

    # Step 8 — Memory-based learning: load history & build few-shot context
    history = load_memory()
    # Keep memory file self-healing so legacy incorrect reward values are fixed.
    save_memory(history)
    few_shot = best_examples(history)
    prompt = generate_prompt(state["document"], few_shot, policy_criteria)

    # Step 9 — LLM call (or deterministic fallback for reproducible baselines)
    if use_mock_model:
        answer = _mock_llm_answer(policy_fit=policy_fit, extracted_data=extracted_data)
    else:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert insurance analyst."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
        )
        answer = response.choices[0].message.content
    parsed = parse_llm_decision(answer)

    # Step 10 — Reward scoring via OpenEnv
    _, reward, _, _ = env.step(answer)

    # Step 11 — Save to memory (learning effect)
    history.append({"answer": answer, "reward": reward})
    save_memory(history)
    persisted_size = len(history[-20:])

    # Step 12 — Confidence scoring and recommendation (JSON/TRl-like learning)
    consistency = history_consistency_score(
        claims_history,
        detected_insurer or extracted_data.get("insurer", ""),
        parsed["decision"],
    )
    confidence = compute_confidence_score(
        reward=reward,
        fraud_detected=False,
        policy_fit_score=policy_fit.get("score", 0.5),
        llm_format_ok=parsed.get("format_ok", False),
        history_consistency=consistency,
    )
    
    # Apply human feedback calibration (reduce confidence if historical patterns show errors)
    confidence = adjust_confidence_for_feedback(
        original_confidence=confidence,
        feedback_log=feedback_log,
        decision=parsed["decision"],
    )
    feedback_decision = apply_feedback_to_decision(
        predicted_decision=parsed["decision"],
        predicted_confidence=confidence,
        claim_snapshot=build_claim_snapshot(
            {
                "insurer_detected": detected_insurer or "Unknown",
                "fraud_detected": False,
                "fraud_reasons": [],
                "structured_claim_data": extracted_data,
            }
        ),
        feedback_log=feedback_log,
    )
    final_decision = feedback_decision.get("decision", parsed["decision"])
    confidence = feedback_decision.get("confidence", confidence)
    feedback_applied = feedback_decision.get("applied", False)

    final_reason = parsed["reason"]
    decision_output = answer
    if feedback_applied:
        final_reason = (
            "Final decision updated from similar prior human feedback. "
            f"Original model decision was {parsed['decision']}. "
            f"{feedback_decision.get('summary', '')} "
            f"Original reason: {parsed['reason']}"
        )
        decision_output = f"Decision: {final_decision}\nReason: {final_reason}"
    
    claim_will_be_approved = final_decision == "Approved" and confidence >= 0.7

    # Step 13 — Build result with insurer, confidence, and structured JSON
    result = {
        "insurer_detected": extracted_data.get("insurer") or detected_insurer or "Unknown",
        "policy_criteria_applied": bool(policy_criteria),
        "fraud_detected": False,
        "rule_engine_fraud_detected": False,
        "fraud_reasons": [],
        "decision_output": decision_output,
        "parsed_decision": final_decision,
        "parsed_reason": final_reason,
        "model_decision": parsed["decision"],
        "reward": reward,
        "memory_size": persisted_size,
        "is_document_correct": True,
        "claim_will_be_approved": claim_will_be_approved,
        "confidence_score": confidence,
        "feedback_override_applied": feedback_applied,
        "feedback_override_summary": feedback_decision.get("summary", ""),
        "final_recommendation": final_recommendation(
            final_decision, confidence, False, final_reason
        ),
        "structured_claim_data": extracted_data,
        "policy_fit": policy_fit,
        "history_consistency": consistency,
    }
    claims_history.append(result)
    save_claims_history(claims_history)
    return result


def format_result(result: dict) -> str:
    """Pretty-format a result dict for terminal / UI display."""
    if "error" in result:
        return f"ERROR: {result['error']}"
    
    # Detect insurer for feedback command (use ASCII-safe characters)
    insurer = result.get('insurer_detected', 'Unknown')
    feedback_hint = "\n[FEEDBACK] Incorrect decision? Run: python feedback_cli.py add"
    feedback_section = ""
    if result.get("feedback_override_applied"):
        feedback_section = (
            "\nFeedback Learning : [APPLIED]\n"
            f"Feedback Summary : {result.get('feedback_override_summary', '')}\n"
            f"Original Model Decision : {result.get('model_decision', result.get('parsed_decision', 'Unknown'))}\n"
        )
    
    # Check for fraud detection
    if result.get("fraud_detected"):
        fraud_section = "[!] FRAUD DETECTED:\n"
        fraud_section += "\n".join([f"  {reason}" for reason in result.get("fraud_reasons", [])])
        return (
            "===== RESULT =====\n"
            f"{fraud_section}\n\n"
            f"{result['decision_output']}\n"
            f"{feedback_section}"
            f"\nConfidence : {result.get('confidence_score', 0.0)}\n"
            f"Will Claim Pass?: No\n"
            f"Recommendation : {result.get('final_recommendation', 'Reject claim.')}\n"
            f"\nExtracted JSON:\n{json.dumps(result.get('structured_claim_data', {}), indent=2)}\n"
            f"\nReward : {result['reward']} (FRAUDULENT DOCUMENT)\n"
            f"{feedback_hint}"
        )
    
    # Normal result formatting
    criteria_info = (
        "[OK] Policy criteria matched"
        if result.get("policy_criteria_applied")
        else "(No criteria file)"
    )
    fraud_check_text = "[OK] Passed"
    if result.get("rule_engine_fraud_detected") and not result.get("fraud_detected"):
        fraud_check_text = "[OVERRIDDEN] Rule engine flagged this file, but prior feedback changed the final decision"

    return (
        "===== RESULT =====\n"
        f"Insurer Detected: {insurer}\n"
        f"Criteria Applied: {criteria_info}\n"
        f"Fraud Check: {fraud_check_text}\n"
        f"{feedback_section}"
        f"Confidence : {result.get('confidence_score', 0.0)}\n"
        f"Document Correct?: {'Yes' if result.get('is_document_correct') else 'No'}\n"
        f"Will Claim Pass?: {'Yes' if result.get('claim_will_be_approved') else 'No'}\n"
        f"Recommendation : {result.get('final_recommendation', 'Needs manual review')}\n"
        f"\n{result['decision_output']}\n"
        f"\nPolicy Fit Score : {result.get('policy_fit', {}).get('score', 0.0)}\n"
        f"History Consistency : {result.get('history_consistency', 0.0)}\n"
        f"\nExtracted JSON:\n{json.dumps(result.get('structured_claim_data', {}), indent=2)}\n"
        f"\nReward : {result['reward']}\n"
        f"Memory : {result['memory_size']} past decision(s) stored\n"
        f"{feedback_hint}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    result = run_inference(pdf_file)
    print("\n" + format_result(result))
