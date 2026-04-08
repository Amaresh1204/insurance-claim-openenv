"""Human-in-the-loop feedback system for continuous model improvement."""

import json
from pathlib import Path
from datetime import datetime


FEEDBACK_FILE = Path(__file__).parent / "feedback_log.json"
CALIBRATION_FILE = Path(__file__).parent / "confidence_calibration.json"


def load_feedback_log() -> list:
    """Load all user feedback records."""
    if FEEDBACK_FILE.exists():
        try:
            return json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_feedback_log(feedback_log: list) -> None:
    """Persist feedback log (keep latest 500 for learning)."""
    FEEDBACK_FILE.write_text(
        json.dumps(feedback_log[-500:], indent=2),
        encoding="utf-8",
    )


def _normalize_text(value: str) -> str:
    """Normalize text values for similarity matching."""
    return " ".join((value or "").strip().lower().split())


def build_claim_snapshot(claim_result: dict) -> dict:
    """Create a compact, comparable snapshot of a claim result."""
    result = claim_result or {}
    structured = result.get("structured_claim_data", {}) or {}
    policy_number = structured.get("policy_number", "")
    fraud_reasons = result.get("fraud_reasons", []) or []

    return {
        "insurer_detected": _normalize_text(
            result.get("insurer_detected") or structured.get("insurer", "")
        ),
        "policy_type": _normalize_text(structured.get("policy_type", "")),
        "procedure": _normalize_text(structured.get("procedure", "")),
        "diagnosis": _normalize_text(structured.get("diagnosis", "")),
        "treatment": _normalize_text(structured.get("treatment", "")),
        "hospital": _normalize_text(structured.get("hospital", "")),
        "policy_number_prefix": _normalize_text(
            policy_number.split("-", 1)[0] if policy_number else ""
        ),
        "claim_amount": structured.get("claim_amount"),
        "hospitalization_days": structured.get("hospitalization_days"),
        "fraud_detected": bool(result.get("fraud_detected", False)),
        "fraud_reason_hint": _normalize_text(fraud_reasons[0] if fraud_reasons else ""),
    }


def _numbers_close(left, right) -> bool:
    """Return True when two numeric values are close enough to be similar."""
    if left is None or right is None:
        return False
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False

    tolerance = max(1.0, max(abs(left_value), abs(right_value)) * 0.15)
    return abs(left_value - right_value) <= tolerance


def _score_feedback_similarity(current_snapshot: dict, stored_snapshot: dict) -> float:
    """Score how similar two claim snapshots are."""
    if not stored_snapshot:
        return 0.0

    weights = {
        "fraud_detected": 3,
        "insurer_detected": 3,
        "policy_type": 2,
        "procedure": 3,
        "diagnosis": 3,
        "treatment": 2,
        "hospital": 1,
        "policy_number_prefix": 2,
        "fraud_reason_hint": 2,
        "claim_amount": 1,
        "hospitalization_days": 1,
    }

    score = 0
    max_score = 0

    for field, weight in weights.items():
        current_value = current_snapshot.get(field)
        stored_value = stored_snapshot.get(field)
        if current_value in (None, "") or stored_value in (None, ""):
            continue

        max_score += weight

        if field in {"claim_amount", "hospitalization_days"}:
            matched = _numbers_close(current_value, stored_value)
        elif field == "fraud_detected":
            matched = bool(current_value) == bool(stored_value)
        else:
            current_text = _normalize_text(str(current_value))
            stored_text = _normalize_text(str(stored_value))
            matched = (
                current_text == stored_text
                or current_text in stored_text
                or stored_text in current_text
            )

        if matched:
            score += weight

    if max_score == 0:
        return 0.0
    return round(score / max_score, 2)


def find_similar_feedback(
    claim_snapshot: dict,
    feedback_log: list,
    predicted_decision: str,
    predicted_confidence: float,
    min_similarity: float = 0.65,
) -> list:
    """Find prior feedback records similar to the current claim."""
    matches = []

    for feedback in feedback_log:
        stored_snapshot = feedback.get("claim_snapshot") or {}
        similarity = _score_feedback_similarity(claim_snapshot, stored_snapshot)

        # Backward-compatible fallback for older feedback entries without snapshots.
        if similarity == 0.0 and not stored_snapshot:
            same_decision = feedback.get("original_decision") == predicted_decision
            close_confidence = (
                abs(feedback.get("original_confidence", 0.5) - predicted_confidence)
                <= 0.1
            )
            if same_decision and close_confidence:
                similarity = 0.7

        if similarity >= min_similarity:
            matched_feedback = dict(feedback)
            matched_feedback["similarity"] = similarity
            matches.append(matched_feedback)

    return sorted(
        matches,
        key=lambda item: (item.get("similarity", 0.0), item.get("timestamp", "")),
        reverse=True,
    )


def apply_feedback_to_decision(
    predicted_decision: str,
    predicted_confidence: float,
    claim_snapshot: dict,
    feedback_log: list,
) -> dict:
    """Use similar prior feedback to influence the current decision."""
    matches = find_similar_feedback(
        claim_snapshot,
        feedback_log,
        predicted_decision,
        predicted_confidence,
    )
    if not matches:
        return {
            "applied": False,
            "decision": predicted_decision,
            "confidence": predicted_confidence,
            "matched_feedback_count": 0,
            "support_ratio": 0.0,
            "strongest_similarity": 0.0,
            "summary": "No similar prior feedback matched this claim.",
        }

    weighted_votes = {"Approved": 0.0, "Rejected": 0.0}
    for match in matches:
        actual_decision = match.get("actual_decision")
        if actual_decision in weighted_votes:
            weighted_votes[actual_decision] += match.get("similarity", 0.0)

    suggested_decision = max(weighted_votes, key=weighted_votes.get)
    total_vote = weighted_votes["Approved"] + weighted_votes["Rejected"]
    support_ratio = round(
        weighted_votes[suggested_decision] / total_vote, 2
    ) if total_vote else 0.0
    strongest_similarity = matches[0].get("similarity", 0.0)
    matched_feedback_count = len(matches)

    should_override = (
        suggested_decision != predicted_decision
        and support_ratio >= 0.6
        and (
            strongest_similarity >= 0.85
            or matched_feedback_count >= 3
        )
    )

    adjusted_confidence = predicted_confidence
    if should_override:
        adjusted_confidence = round(
            min(0.98, max(predicted_confidence, 0.7 + (0.2 * support_ratio))),
            2,
        )
    elif suggested_decision == predicted_decision and support_ratio >= 0.6:
        adjusted_confidence = round(
            min(0.98, max(predicted_confidence, predicted_confidence + 0.05)),
            2,
        )

    summary = (
        f"Matched {matched_feedback_count} similar feedback record(s); "
        f"strongest similarity={strongest_similarity:.2f}; "
        f"historical decision={suggested_decision}; support={support_ratio:.2f}."
    )

    return {
        "applied": should_override,
        "decision": suggested_decision if should_override else predicted_decision,
        "confidence": adjusted_confidence,
        "matched_feedback_count": matched_feedback_count,
        "support_ratio": support_ratio,
        "strongest_similarity": strongest_similarity,
        "summary": summary,
    }


def add_feedback(
    claim_id: str,
    original_decision: str,
    original_confidence: float,
    actual_decision: str,
    feedback_reason: str,
    feedback_details: str = "",
    claim_snapshot: dict = None,
) -> dict:
    """Record user feedback on a claim decision.
    
    Args:
        claim_id: Unique claim identifier
        original_decision: What AI decided (Approved/Rejected)
        original_confidence: AI's confidence score
        actual_decision: What user says is correct (Approved/Rejected)
        feedback_reason: Why user is correcting (e.g., 'fraud_detected', 'wrong_criteria', 'documentation_issue')
        feedback_details: Additional explanation
    
    Returns:
        Feedback record dict
    """
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "claim_id": claim_id,
        "original_decision": original_decision,
        "original_confidence": original_confidence,
        "actual_decision": actual_decision,
        "is_false_positive": (
            original_decision == "Approved" and actual_decision == "Rejected"
        ),
        "is_false_negative": (
            original_decision == "Rejected" and actual_decision == "Approved"
        ),
        "is_correct": original_decision == actual_decision,
        "feedback_reason": feedback_reason,
        "feedback_details": feedback_details,
        "claim_snapshot": claim_snapshot or {},
    }
    
    log = load_feedback_log()
    log.append(feedback)
    save_feedback_log(log)
    
    return feedback


def compute_calibration_stats() -> dict:
    """Analyze feedback to compute confidence calibration."""
    log = load_feedback_log()
    
    if not log:
        return {
            "total_feedback": 0,
            "accuracy": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "confidence_bins": {},
            "top_false_positive_reasons": [],
            "top_false_negative_reasons": [],
        }
    
    correct_count = sum(1 for f in log if f.get("is_correct"))
    fp_count = sum(1 for f in log if f.get("is_false_positive"))
    fn_count = sum(1 for f in log if f.get("is_false_negative"))
    
    accuracy = correct_count / len(log) if log else 0.0
    fp_rate = fp_count / len(log) if log else 0.0
    fn_rate = fn_count / len(log) if log else 0.0
    
    # Bin confidences to see calibration
    conf_bins = {
        "0.0-0.3": [],
        "0.3-0.5": [],
        "0.5-0.7": [],
        "0.7-0.9": [],
        "0.9-1.0": [],
    }
    for f in log:
        conf = f.get("original_confidence", 0.5)
        if conf < 0.3:
            conf_bins["0.0-0.3"].append(f)
        elif conf < 0.5:
            conf_bins["0.3-0.5"].append(f)
        elif conf < 0.7:
            conf_bins["0.5-0.7"].append(f)
        elif conf < 0.9:
            conf_bins["0.7-0.9"].append(f)
        else:
            conf_bins["0.9-1.0"].append(f)
    
    bin_stats = {
        k: {
            "count": len(v),
            "accuracy": sum(1 for x in v if x.get("is_correct")) / len(v) if v else 0.0,
        }
        for k, v in conf_bins.items()
    }
    
    # Top false positive reasons
    fp_reasons = {}
    for f in log:
        if f.get("is_false_positive"):
            reason = f.get("feedback_reason", "unknown")
            fp_reasons[reason] = fp_reasons.get(reason, 0) + 1
    top_fp = sorted(fp_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Top false negative reasons
    fn_reasons = {}
    for f in log:
        if f.get("is_false_negative"):
            reason = f.get("feedback_reason", "unknown")
            fn_reasons[reason] = fn_reasons.get(reason, 0) + 1
    top_fn = sorted(fn_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "total_feedback": len(log),
        "correct_decisions": correct_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "overall_accuracy": round(accuracy, 3),
        "false_positive_rate": round(fp_rate, 3),
        "false_negative_rate": round(fn_rate, 3),
        "confidence_bins": bin_stats,
        "top_false_positive_reasons": top_fp,
        "top_false_negative_reasons": top_fn,
    }


def adjust_confidence_for_feedback(
    original_confidence: float,
    feedback_log: list,
    decision: str,
) -> float:
    """Adjust confidence score based on historical feedback patterns."""
    if not feedback_log:
        return original_confidence
    
    # Find similar confidence ranges in feedback to adjust
    similar_confident = [
        f
        for f in feedback_log
        if abs(f.get("original_confidence", 0.5) - original_confidence) < 0.1
        and f.get("original_decision") == decision
    ]
    
    if not similar_confident:
        return original_confidence
    
    accuracy_in_bin = sum(1 for f in similar_confident if f.get("is_correct")) / len(
        similar_confident
    )
    
    # If accuracy is low in this confidence bin, reduce confidence
    if accuracy_in_bin < 0.7:
        adjustment = 0.9 - (0.7 - accuracy_in_bin) * 0.3
        adjusted = original_confidence * adjustment
        return round(max(0.0, min(1.0, adjusted)), 2)
    
    return original_confidence


def generate_feedback_summary() -> str:
    """Generate a human-readable feedback summary."""
    stats = compute_calibration_stats()
    
    if stats["total_feedback"] == 0:
        return "No feedback recorded yet. Start providing feedback to improve model calibration."
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║               SYSTEM CALIBRATION FEEDBACK REPORT               ║
╚════════════════════════════════════════════════════════════════╝

📊 Overall Accuracy:
  • Correct Decisions: {stats['correct_decisions']}/{stats['total_feedback']} ({stats['overall_accuracy']*100:.1f}%)
  • False Positives: {stats['false_positives']} ({stats['false_positive_rate']*100:.1f}%)
  • False Negatives: {stats['false_negatives']} ({stats['false_negative_rate']*100:.1f}%)

📈 Confidence Bin Calibration:
"""
    
    for bin_range, bin_stats in stats["confidence_bins"].items():
        if bin_stats["count"] > 0:
            summary += f"  • [{bin_range}] Accuracy: {bin_stats['accuracy']*100:.1f}% (n={bin_stats['count']})\n"
    
    if stats["top_false_positive_reasons"]:
        summary += f"\n⚠️  Top False Positive Reasons (should be Rejected but AI approved):\n"
        for reason, count in stats["top_false_positive_reasons"]:
            summary += f"  • {reason}: {count} cases\n"
    
    if stats["top_false_negative_reasons"]:
        summary += f"\n⚠️  Top False Negative Reasons (should be Approved but AI rejected):\n"
        for reason, count in stats["top_false_negative_reasons"]:
            summary += f"  • {reason}: {count} cases\n"

    summary += (
        "\n[INFO] How feedback is used right now:\n"
        "  • Yes: feedback is already used during inference to calibrate confidence scores.\n"
        "  • Yes: repeated mistakes in a confidence range reduce future confidence for similar decisions.\n"
        "  • No: the base LLM is not automatically retrained or fine-tuned by this app.\n"
        "\n[ACTION] If you want true model retraining, export feedback_log.json and use it as supervised training data in a separate fine-tuning pipeline.\n"
    )
    return summary
