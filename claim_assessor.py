"""Structured extraction, policy checks, and confidence scoring for claims."""

import json
import re
from datetime import datetime
from pathlib import Path


CLAIMS_HISTORY_FILE = Path(__file__).parent / "claims_history.json"


def _to_number(value: str):
    cleaned = re.sub(r"[^0-9.]", "", (value or ""))
    if not cleaned:
        return None
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return None


def _parse_date(value: str):
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def extract_structured_claim_data(document_text: str) -> dict:
    """Extract structured key-value data from the PDF text."""
    data = {
        "insurer": "",
        "policy_type": "",
        "patient": "",
        "age": None,
        "policy_number": "",
        "procedure": "",
        "hospital": "",
        "admission_date": "",
        "discharge_date": "",
        "coverage": "",
        "sum_insured": None,
        "claim_amount": None,
        "claim_filed_on": "",
        "doctor": "",
        "diagnosis": "",
        "treatment": "",
        "notes": "",
    }

    key_map = {
        "insurer": "insurer",
        "policy type": "policy_type",
        "patient": "patient",
        "age": "age",
        "policy number": "policy_number",
        "procedure": "procedure",
        "hospital": "hospital",
        "date of admission": "admission_date",
        "date of discharge": "discharge_date",
        "coverage": "coverage",
        "sum insured": "sum_insured",
        "claim amount": "claim_amount",
        "claim filed on": "claim_filed_on",
        "doctor": "doctor",
        "diagnosis": "diagnosis",
        "treatment": "treatment",
        "notes": "notes",
    }

    for raw_line in document_text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = raw_key.strip().lower()
        value = raw_value.strip()

        if key in key_map:
            field = key_map[key]
            if field in {"age", "sum_insured", "claim_amount"}:
                data[field] = _to_number(value)
            else:
                data[field] = value

    admission = _parse_date(data.get("admission_date", ""))
    discharge = _parse_date(data.get("discharge_date", ""))
    if admission and discharge and discharge >= admission:
        data["hospitalization_days"] = (discharge - admission).days + 1
    else:
        data["hospitalization_days"] = None

    return data


def evaluate_policy_fit(claim_data: dict, policy_criteria: dict) -> dict:
    """Evaluate extracted claim data against insurer policy criteria."""
    checks = []

    if not policy_criteria:
        return {
            "score": 0.5,
            "checks": [
                {
                    "name": "policy_criteria_available",
                    "passed": False,
                    "detail": "No matching policy criteria file was found.",
                    "weight": 1,
                }
            ],
            "passed_checks": 0,
            "total_checks": 1,
        }

    criteria = policy_criteria.get("claim_approval_criteria", {})

    # Check claim amount vs sum insured
    claim_amount = claim_data.get("claim_amount")
    sum_insured = claim_data.get("sum_insured")
    if claim_amount is not None and sum_insured is not None:
        passed = claim_amount <= sum_insured
        checks.append(
            {
                "name": "sum_insured_available",
                "passed": passed,
                "detail": f"Claim amount ({claim_amount}) <= Sum insured ({sum_insured})",
                "weight": 3,
            }
        )

    # Check minimum hospitalization hours
    min_hours = criteria.get("minimum_hospitalization_hours")
    days = claim_data.get("hospitalization_days")
    if min_hours is not None and days is not None:
        actual_hours = days * 24
        passed = actual_hours >= min_hours
        checks.append(
            {
                "name": "minimum_hospitalization_hours",
                "passed": passed,
                "detail": f"Hospitalization hours ({actual_hours}) >= Required ({min_hours})",
                "weight": 2,
            }
        )

    # Check critical field presence
    for field in ["insurer", "policy_number", "hospital", "procedure", "diagnosis"]:
        passed = bool(claim_data.get(field))
        checks.append(
            {
                "name": f"field_{field}_present",
                "passed": passed,
                "detail": f"Field '{field}' present",
                "weight": 1,
            }
        )

    if not checks:
        return {"score": 0.5, "checks": [], "passed_checks": 0, "total_checks": 0}

    total_weight = sum(c["weight"] for c in checks)
    passed_weight = sum(c["weight"] for c in checks if c["passed"])
    score = round(passed_weight / total_weight, 2) if total_weight else 0.5

    return {
        "score": score,
        "checks": checks,
        "passed_checks": sum(1 for c in checks if c["passed"]),
        "total_checks": len(checks),
    }


def parse_llm_decision(answer: str) -> dict:
    """Parse decision and reason from LLM output."""
    text = (answer or "").strip()
    decision_match = re.search(r"decision\s*:\s*(approved|rejected)", text, re.IGNORECASE)
    reason_match = re.search(r"reason\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    decision = decision_match.group(1).capitalize() if decision_match else "Unknown"
    reason = reason_match.group(1).strip() if reason_match else ""

    return {
        "decision": decision,
        "reason": reason,
        "format_ok": bool(decision_match and reason_match),
    }


def load_claims_history() -> list:
    if CLAIMS_HISTORY_FILE.exists():
        try:
            return json.loads(CLAIMS_HISTORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_claims_history(history: list) -> None:
    CLAIMS_HISTORY_FILE.write_text(
        json.dumps(history[-100:], indent=2),
        encoding="utf-8",
    )


def history_consistency_score(history: list, insurer: str, decision: str) -> float:
    """How consistent this decision is with past similar claims."""
    if decision not in {"Approved", "Rejected"}:
        return 0.4

    similar = [
        h
        for h in history
        if (h.get("insurer_detected") or "").lower() == (insurer or "").lower()
        and not h.get("fraud_detected", False)
        and h.get("parsed_decision") in {"Approved", "Rejected"}
    ]
    if not similar:
        return 0.5

    match_count = sum(1 for h in similar if h.get("parsed_decision") == decision)
    return round(match_count / len(similar), 2)


def compute_confidence_score(
    reward: float,
    fraud_detected: bool,
    policy_fit_score: float,
    llm_format_ok: bool,
    history_consistency: float,
) -> float:
    """Compute a confidence score in [0, 1]."""
    if fraud_detected:
        return 0.99

    score = 0.2
    score += 0.3 * max(0.0, min(1.0, reward))
    score += 0.3 * max(0.0, min(1.0, policy_fit_score))
    score += 0.1 * (1.0 if llm_format_ok else 0.0)
    score += 0.1 * max(0.0, min(1.0, history_consistency))

    return round(max(0.0, min(1.0, score)), 2)


def final_recommendation(decision: str, confidence: float, fraud_detected: bool, reason: str = "") -> str:
    """Return user-facing claim recommendation."""
    if fraud_detected:
        rec = "Reject claim (fraud risk detected)."
    elif decision == "Approved" and confidence >= 0.7:
        rec = "Claim likely to be approved."
    elif decision == "Rejected":
        rec = "Claim likely to be rejected."
    else:
        rec = "Needs manual review before final claim decision."
        
    if reason:
        return f"{rec}<br>Reason: {reason}"
    return rec
