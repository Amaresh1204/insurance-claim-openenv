"""Gradio UI for the Insurance Claim AI system with built-in feedback.

Features:
- Process insurance claims from PDFs
- Provide feedback on AI decisions
- View model calibration reports
- Learn from corrections

Run: python app.py
"""

import json
import gradio as gr
from pathlib import Path
from uuid import uuid4
from claim_pipeline import run_inference
from feedback_handler import (
    add_feedback,
    build_claim_snapshot,
    compute_calibration_stats,
)
from ui import create_ui

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
last_result = None
claims_history_file = Path(__file__).parent / "claims_history.json"
feedback_log_file = Path(__file__).parent / "feedback_log.json"
memory_file = Path(__file__).parent / "memory.json"
ui_episode_id = str(uuid4())
ui_step_count = 0
ui_total_reward = 0.0


# ---------------------------------------------------------------------------
# Backend helpers — dashboard-style builders
# ---------------------------------------------------------------------------

_EMPTY = ""  # sentinel for blank Markdown outputs


# ---------------------------------------------------------------------------
# Dashboard builder helpers
# ---------------------------------------------------------------------------

def _confidence_bar(confidence: float) -> str:
    """Unicode progress bar, e.g. ████████░░ 80%."""
    pct = int(confidence * 100)
    filled = round(confidence * 10)
    empty = 10 - filled
    full_block = chr(0x2588)
    light_block = chr(0x2591)
    bar = (full_block * filled) + (light_block * empty)
    return f"`{bar}` **{pct}%**"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "\U0001f7e2 High"
    elif confidence >= 0.4:
        return "\U0001f7e1 Medium"
    return "\U0001f534 Low"


def _build_decision_hero(result: dict) -> str:
    decision = result.get("parsed_decision", "Unknown")
    confidence = result.get("confidence_score", 0.0)
    label = _confidence_label(confidence)
    pct = int(confidence * 100)
    is_approved = decision == "Approved"
    bg = "linear-gradient(135deg,#dcfce7 0%,#bbf7d0 100%)" if is_approved else "linear-gradient(135deg,#fee2e2 0%,#fecaca 100%)"
    border = "#86efac" if is_approved else "#fca5a5"
    title_color = "#15803d" if is_approved else "#dc2626"
    sub_color = "#166534" if is_approved else "#991b1b"
    badge_bg = "#15803d" if is_approved else "#dc2626"
    conf_color = "#16a34a" if pct >= 80 else ("#f59e0b" if pct >= 50 else "#ef4444")
    icon = "&#10003;" if is_approved else "&#10007;"
    label_text = label.split()[-1] if label else ""
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:20px 24px;">'
        f'<div style="font-size:24px;font-weight:800;color:{title_color};margin-bottom:4px;">{icon} {decision.upper()}</div>'
        f'<div style="color:{sub_color};font-size:13px;margin-bottom:14px;">AI Decision Result</div>'
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
        f'<span style="background:{badge_bg};color:white;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:600;">Confidence: {pct}%</span>'
        f'<span style="background:{conf_color};color:white;padding:4px 14px;border-radius:20px;font-size:13px;">{label_text}</span>'
        f'</div></div>'
    )


def _build_quick_insights(result: dict) -> str:
    criteria = result.get("policy_criteria_applied", False)
    fraud = result.get("fraud_detected", False)
    confidence = result.get("confidence_score", 0.0)
    doc_correct = result.get("is_document_correct", False)
    auto_ok = confidence >= 0.7 and doc_correct and not fraud

    def badge(text, ok):
        bg = "#dcfce7" if ok else "#fef9c3"
        color = "#166534" if ok else "#854d0e"
        return f'<span style="background:{bg};color:{color};padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;display:inline-block;">{text}</span>'

    pm = badge("&#10004; Policy Match" if criteria else "&#9888; No Policy Match", criteria)
    fd = badge("&#10003; No Fraud" if not fraud else "&#9888; Fraud Detected", not fraud)
    rv = badge("&#10003; Auto Decision OK" if auto_ok else "&#9888; Manual Review Required", auto_ok)
    return f'<div style="display:flex;gap:8px;flex-wrap:wrap;padding:8px 0;">{pm}{fd}{rv}</div>'


def _build_metrics_col1(result: dict) -> str:
    insurer = result.get("insurer_detected", "Unknown")
    pf = result.get("policy_fit", {}).get("score", 0.0)
    return (
        '<div style="border:1px solid #e2e8f0;border-radius:12px;padding:16px 18px;background:#f8fafc;">'
        '<div style="color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Insurer</div>'
        f'<div style="font-size:20px;font-weight:700;color:#0f172a;margin-bottom:14px;">{insurer}</div>'
        '<div style="color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Policy Fit Score</div>'
        f'<div style="font-size:20px;font-weight:700;color:#0f172a;">{pf}</div>'
        '</div>'
    )


def _build_metrics_col2(result: dict) -> str:
    hc = result.get("history_consistency", 0.0)
    rw = result.get("reward", 0.0)
    return (
        '<div style="border:1px solid #e2e8f0;border-radius:12px;padding:16px 18px;background:#f8fafc;">'
        '<div style="color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">History Consistency</div>'
        f'<div style="font-size:20px;font-weight:700;color:#0f172a;margin-bottom:14px;">{hc}</div>'
        '<div style="color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Reward Score</div>'
        f'<div style="font-size:20px;font-weight:700;color:#0f172a;">{rw}</div>'
        '</div>'
    )


def _build_metrics_col3(result: dict) -> str:
    mem = result.get("memory_size", 0)
    doc_correct = result.get("is_document_correct", False)
    claim_pass = result.get("claim_will_be_approved", False)
    d = '<span style="color:#16a34a;font-weight:700;">Yes</span>' if doc_correct else '<span style="color:#dc2626;font-weight:700;">No</span>'
    p = '<span style="color:#16a34a;font-weight:700;">Yes</span>' if claim_pass else '<span style="color:#dc2626;font-weight:700;">No</span>'
    return (
        '<div style="border:1px solid #e2e8f0;border-radius:12px;padding:16px 18px;background:#f8fafc;">'
        '<div style="color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Memory</div>'
        f'<div style="font-size:20px;font-weight:700;color:#0f172a;margin-bottom:14px;">{mem} past decision(s)</div>'
        '<div style="color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Validation</div>'
        f'<div style="font-size:13px;color:#334155;">Doc Valid: {d} &nbsp;&nbsp; Claim Passes: {p}</div>'
        '</div>'
    )


def _build_fraud_md(result: dict) -> str:
    fraud = result.get("fraud_detected", False)
    reasons = result.get("fraud_reasons", [])
    if not fraud:
        return (
            '<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:14px 18px;display:flex;gap:12px;align-items:flex-start;">'
            '<div style="width:10px;height:10px;border-radius:50%;background:#16a34a;margin-top:4px;flex-shrink:0;"></div>'
            '<div><div style="font-weight:700;color:#15803d;font-size:14px;">Fraud Status: CLEAN</div>'
            '<div style="color:#166534;font-size:13px;margin-top:3px;">No fraud indicators detected.</div></div></div>'
        )
    items_html = "".join(f'<li style="margin-bottom:4px;">{r}</li>' for r in reasons)
    return (
        '<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;padding:14px 18px;">'
        '<div style="font-weight:700;color:#dc2626;font-size:14px;margin-bottom:8px;">&#9888; Fraud Status: RISK DETECTED</div>'
        f'<ul style="margin:0;padding-left:18px;color:#991b1b;font-size:13px;">{items_html}</ul></div>'
    )


def _build_recommendation_md(result: dict) -> str:
    rec = result.get("final_recommendation", "Needs manual review")
    confidence = result.get("confidence_score", 0.0)
    border_color = "#f59e0b" if confidence >= 0.7 else "#ef4444"
    title_color = "#92400e" if confidence >= 0.7 else "#991b1b"
    text_color = "#78350f" if confidence >= 0.7 else "#7f1d1d"
    bg = "#fffbeb" if confidence >= 0.7 else "#fef2f2"
    return (
        f'<div style="background:{bg};border-left:4px solid {border_color};border-radius:0 8px 8px 0;padding:14px 18px;">'
        f'<div style="font-weight:700;color:{title_color};font-size:14px;margin-bottom:6px;">Recommendation</div>'
        f'<div style="color:{text_color};font-size:13px;font-style:italic;">{rec}</div></div>'
    )


def _build_reason_md(result: dict) -> str:
    reason = result.get("parsed_reason", "")
    if not reason:
        return '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 18px;color:#94a3b8;font-style:italic;">No reason provided.</div>'
    parts = [s.strip().rstrip(".") for s in reason.replace(". ", ".\n").split("\n") if s.strip()]
    if len(parts) > 1:
        items_html = "".join(f'<li style="margin-bottom:6px;color:#334155;font-size:13px;">{s}.</li>' for s in parts if s)
        body = f'<ul style="margin:8px 0 0 0;padding-left:18px;">{items_html}</ul>'
    else:
        body = f'<p style="margin:8px 0 0 0;color:#334155;font-size:13px;">{reason}</p>'
    return (
        '<div style="background:#f1f5f9;border-left:4px solid #6366f1;border-radius:0 10px 10px 0;padding:14px 18px;">'
        '<div style="font-weight:700;color:#4338ca;font-size:14px;margin-bottom:4px;">Why This Decision?</div>'
        f'{body}</div>'
    )


def _build_checks_summary(result: dict) -> str:
    checks = result.get("policy_fit", {}).get("checks", [])
    if not checks:
        return _EMPTY
    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)
    icon = "\u2714" if passed == total else ("\u26a0\ufe0f" if passed > 0 else "\u274c")
    return f"{icon} **{passed} / {total} checks passed**"


_FRIENDLY_CHECK_NAMES = {
    "sum_insured_available": "Claim Within Coverage Limit",
    "minimum_hospitalization_hours": "Minimum Hospitalization Period",
    "field_insurer_present": "Insurer Name Provided",
    "field_policy_number_present": "Policy Number Provided",
    "field_hospital_present": "Hospital Name Provided",
    "field_procedure_present": "Procedure Details Provided",
    "field_diagnosis_present": "Diagnosis Information Provided",
    "policy_criteria_available": "Policy Criteria On File",
}

_FRIENDLY_DETAIL_TEMPLATES = {
    "sum_insured_available": lambda c: c["detail"]
        .replace("Claim amount", "Claim Amount")
        .replace("<= Sum insured", "is within the Sum Insured of"),
    "minimum_hospitalization_hours": lambda c: c["detail"]
        .replace("Hospitalization hours", "Total Stay")
        .replace(">= Required", "meets the minimum required"),
    "field_insurer_present": lambda c: "Insurer name was found in the document" if c["passed"] else "Insurer name is missing from the document",
    "field_policy_number_present": lambda c: "Policy number was found in the document" if c["passed"] else "Policy number is missing from the document",
    "field_hospital_present": lambda c: "Hospital name was found in the document" if c["passed"] else "Hospital name is missing from the document",
    "field_procedure_present": lambda c: "Procedure details were found in the document" if c["passed"] else "Procedure details are missing from the document",
    "field_diagnosis_present": lambda c: "Diagnosis information was found in the document" if c["passed"] else "Diagnosis information is missing from the document",
}

_WEIGHT_LABELS = {1: "Standard", 2: "Important", 3: "Critical"}


def _build_checks_md(result: dict) -> str:
    checks = result.get("policy_fit", {}).get("checks", [])
    if not checks:
        return "*No policy checks available.*"
    rows = ""
    for c in checks:
        icon = "\u2714" if c["passed"] else "\u274c"
        friendly_name = _FRIENDLY_CHECK_NAMES.get(c["name"], c["name"])
        detail_fn = _FRIENDLY_DETAIL_TEMPLATES.get(c["name"])
        friendly_detail = detail_fn(c) if detail_fn else c["detail"]
        weight_label = _WEIGHT_LABELS.get(c["weight"], str(c["weight"]))
        rows += f"| {icon} | {friendly_name} | {friendly_detail} | {weight_label} |\n"
    return (
        "| Status | Validation Check | Details | Priority |\n"
        "|--------|-----------------|---------|----------|\n"
        f"{rows}"
    )


def _build_feedback_override_md(result: dict) -> str:
    if not result.get("feedback_override_applied"):
        return _EMPTY
    model_dec = result.get("model_decision", result.get("parsed_decision", "Unknown"))
    summary = result.get("feedback_override_summary", "")
    return (
        "### \U0001f504 Feedback Override Applied\n\n"
        f"Original model decision was **{model_dec}**.\n\n"
        f"{summary}"
    )


def _format_calibration_md(stats: dict) -> str:
    """Convert calibration stats dict into clean sectioned Markdown."""
    if not stats or stats.get("total_feedback", 0) == 0:
        return (
            "## \U0001f4ca No Data Yet\n\n"
            "No feedback recorded. Provide corrections on processed claims "
            "to start building calibration data.\n\n"
            "---\n\n"
            "### \U0001f4a1 How to Start\n\n"
            "1. Process a claim in the **Process Claim** tab.\n"
            "2. Go to **Provide Feedback** and submit a correction.\n"
            "3. Return here and click **Refresh Report**."
        )

    total = stats["total_feedback"]
    correct = stats.get("correct_decisions", 0)
    fp = stats.get("false_positives", 0)
    fn = stats.get("false_negatives", 0)
    accuracy = stats.get("overall_accuracy", 0.0)
    fp_rate = stats.get("false_positive_rate", 0.0)
    fn_rate = stats.get("false_negative_rate", 0.0)

    acc_icon = "\u2705" if accuracy >= 0.8 else ("\u26a0\ufe0f" if accuracy >= 0.6 else "\u274c")
    out = (
        f"## \U0001f4ca Model Calibration Report\n\n"
        f"*Based on {total} feedback record(s)*\n\n"
        f"---\n\n"
        f"### \U0001f4ca Accuracy\n\n"
        f"- Accuracy: **{accuracy * 100:.1f}%** ({correct} / {total} correct) {acc_icon}\n"
        f"- False Positive Rate: **{fp_rate * 100:.1f}%** ({fp} case(s))\n"
        f"- False Negative Rate: **{fn_rate * 100:.1f}%** ({fn} case(s))\n"
    )

    bins = stats.get("confidence_bins", {})
    active_bins = [(k, v) for k, v in bins.items() if v.get("count", 0) > 0]
    if active_bins:
        bin_lines = []
        for rng, bv in active_bins:
            bacc = bv.get("accuracy", 0.0)
            bcount = bv.get("count", 0)
            bicon = "\u2705" if bacc >= 0.8 else ("\u26a0\ufe0f" if bacc >= 0.6 else "\u274c")
            bin_lines.append(
                f"- {bicon} **[{rng}]** \u2014 Accuracy: "
                f"{bacc * 100:.1f}% (n={bcount})"
            )
        out += (
            "\n---\n\n"
            "### \U0001f4c9 Confidence Bin Calibration\n\n"
            + "\n".join(bin_lines) + "\n"
        )

    top_fp = stats.get("top_false_positive_reasons", [])
    top_fn = stats.get("top_false_negative_reasons", [])
    if top_fp or top_fn:
        out += "\n---\n\n### \u26a0\ufe0f Error Analysis\n\n"
        if top_fp:
            total_fp_cases = sum(c for _, c in top_fp)
            out += f"\U0001f534 **Fraud detected incorrectly: {total_fp_cases} case(s)**\n\n"
            out += "\n".join(f"- `{r}`: {c} case(s)" for r, c in top_fp) + "\n\n"
        if top_fn:
            total_fn_cases = sum(c for _, c in top_fn)
            out += f"\U0001f7e0 **Wrong criteria applied: {total_fn_cases} case(s)**\n\n"
            out += "\n".join(f"- `{r}`: {c} case(s)" for r, c in top_fn) + "\n"

    out += (
        "\n---\n\n"
        "### \U0001f4a1 Insights\n\n"
        "- Feedback adjusts confidence calibration at inference time in real-time.\n"
        "- Repeated errors in a confidence range **lower future scores** for that range.\n"
    )
    return out


def process_claim(file):
    """Process a PDF and return all 13 dashboard outputs."""
    global last_result, ui_step_count, ui_total_reward

    if file is None:
        return ("", "", "", "", "", "", "", "", "", "", "", gr.update(visible=False))

    pdf_path = file if isinstance(file, str) else getattr(file, "name", "")
    result = run_inference(pdf_path)
    last_result = result

    if "error" in result:
        err_html = (
            '<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;padding:14px 18px;">'
            f'<div style="font-weight:700;color:#dc2626;">&#10007; Error</div>'
            f'<div style="color:#991b1b;font-size:13px;margin-top:4px;">{result["error"]}</div></div>'
        )
        return (err_html, _EMPTY, _EMPTY, _EMPTY, _EMPTY,
                _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, "",
                gr.update(visible=True))

    ui_step_count += 1
    ui_total_reward += float(result.get("reward", 0.0))
    quick = (
        f"Insurer: {result.get('insurer_detected', 'Unknown')}  |  "
        f"Decision: {result.get('parsed_decision', 'Unknown')}  |  "
        f"Confidence: {result.get('confidence_score', 0.0):.2f}"
    )

    return (
        _build_decision_hero(result),       # 0  decision_hero_md
        _build_quick_insights(result),       # 1  quick_insights_md
        _build_metrics_col1(result),         # 2  metrics_col1_md
        _build_metrics_col2(result),         # 3  metrics_col2_md
        _build_metrics_col3(result),         # 4  metrics_col3_md
        _build_fraud_md(result),             # 5  fraud_md
        _build_recommendation_md(result),    # 6  recommendation_md
        _build_checks_summary(result),       # 7  checks_summary_md
        _build_checks_md(result),            # 8  checks_md
        _build_feedback_override_md(result), # 9  feedback_override_md
        quick,                               # 10 quick_info_bar
        gr.update(visible=True),             # 11 results_col
    )


def clear_outputs():
    """Reset PDF input and all result output fields."""
    return (None, "", "", "", "", "", "", "", "", "", "", "", gr.update(visible=False))


def reset_system():
    """Hard reset app state and persisted learning/history JSON files."""
    global last_result, ui_episode_id, ui_step_count, ui_total_reward

    for path in [
        Path(__file__).parent / "claims_history.json",
        Path(__file__).parent / "feedback_log.json",
        Path(__file__).parent / "memory.json",
    ]:
        path.write_text(json.dumps([], indent=2), encoding="utf-8")

    last_result = None
    ui_episode_id = str(uuid4())
    ui_step_count = 0
    ui_total_reward = 0.0
    
    gr.Info("System was reset")
    
    return (
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "✅ System reset successful. Ready for a new claim.",
        gr.update(visible=False),
    )


def _safe_list_length(path: Path) -> int:
    """Return list length from JSON file, or zero if unreadable/missing."""
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return len(payload) if isinstance(payload, list) else 0


def get_state_snapshot():
    """Return a UI-friendly snapshot of the current environment state."""
    last_observation = None
    if last_result is not None:
        last_observation = {
            "parsed_decision": last_result.get("parsed_decision", "Unknown"),
            "confidence_score": last_result.get("confidence_score", 0.0),
            "fraud_detected": last_result.get("fraud_detected", False),
            "insurer_detected": last_result.get("insurer_detected", "Unknown"),
            "policy_fit_score": last_result.get("policy_fit", {}).get("score", 0.0),
            "reward": last_result.get("reward", 0.0),
        }

    snapshot = {
        "episode_id": ui_episode_id,
        "step_count": ui_step_count,
        "total_reward": round(ui_total_reward, 4),
        "last_observation": last_observation,
        "persisted_counts": {
            "claims_history": _safe_list_length(claims_history_file),
            "feedback_log": _safe_list_length(feedback_log_file),
            "memory": _safe_list_length(memory_file),
        },
    }

    state_md = (
        f"Episode ID: {snapshot['episode_id']}<br>"
        f"Step Count: {snapshot['step_count']}<br>"
        f"Total Reward: {snapshot['total_reward']}<br>"
        f"Claims History Records: {snapshot['persisted_counts']['claims_history']}<br>"
        f"Feedback Records: {snapshot['persisted_counts']['feedback_log']}<br>"
        f"Memory Records: {snapshot['persisted_counts']['memory']}"
    )
    return state_md, snapshot


def submit_feedback(correct_decision, feedback_reason, feedback_details):
    """Record user feedback on the last processed claim."""
    global last_result

    if last_result is None:
        return (
            "&#9888;&#65039; **No claim processed yet.** "
            "Go to the *Process Claim* tab first."
        )

    if not correct_decision:
        return "&#9888;&#65039; **Please select a correct decision.**"

    try:
        original_decision = last_result.get("parsed_decision", "Unknown")
        original_confidence = last_result.get("confidence_score", 0.5)
        claim_id = f"claim_{id(last_result)}"

        decision_map = {
            "Approved": "Approved",
            "Rejected": "Rejected",
        }
        actual_decision = decision_map.get(correct_decision, correct_decision)

        feedback_record = add_feedback(
            claim_id=claim_id,
            original_decision=original_decision,
            original_confidence=original_confidence,
            actual_decision=actual_decision,
            feedback_reason=feedback_reason,
            feedback_details=feedback_details,
            claim_snapshot=build_claim_snapshot(last_result),
        )

        is_error = feedback_record.get("is_false_positive") or feedback_record.get(
            "is_false_negative"
        )

        if is_error:
            error_type = (
                "FALSE POSITIVE \u2014 approved when should be rejected"
                if feedback_record.get("is_false_positive")
                else "FALSE NEGATIVE \u2014 rejected when should be approved"
            )
            status_line = f"\u26a0\ufe0f Error Detected: {error_type}"
        else:
            status_line = "\u2705 Confirmed Correct"

        details_line = f"- **Details:** {feedback_details}\n" if feedback_details else ""
        return (
            f"### \u2705 Feedback Saved\n\n"
            f"---\n\n"
            f"- **Original Decision:** {original_decision}\n"
            f"- **Confidence:** {original_confidence:.0%}\n"
            f"- **Correct Decision:** {actual_decision}\n"
            f"- **Reason:** {feedback_reason}\n"
            f"{details_line}"
            f"\n---\n\n"
            f"*Status:* {status_line}\n\n"
            f"*This feedback helps improve future predictions.*"
        )

    except Exception as e:
        return f"&#10060; **Failed to record feedback:** {e}"


def clear_feedback():
    """Reset feedback form fields."""
    return (
        gr.update(value=None),
        gr.update(value="Policy mismatch"),
        gr.update(value=""),
        "",
    )


def refresh_reports():
    """Load calibration stats and format as Markdown + raw dict."""
    try:
        stats = compute_calibration_stats()
    except Exception as e:
        stats = {"error": str(e)}
    try:
        report_md = _format_calibration_md(stats)
    except Exception as e:
        report_md = f"\u274c Error formatting report: {e}"
    return report_md, stats


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

demo, custom_css = create_ui(
    process_claim=process_claim,
    clear_outputs=clear_outputs,
    reset_system=reset_system,
    get_state_snapshot=get_state_snapshot,
    submit_feedback=submit_feedback,
    clear_feedback=clear_feedback,
    refresh_reports=refresh_reports,
    theme=THEME,
)

if __name__ == "__main__":
    demo.launch(share=True, theme=THEME, css=custom_css)
