#!/usr/bin/env python
"""Interactive CLI to provide feedback on claim decisions."""

import json
import sys
from pathlib import Path
from feedback_handler import (
    add_feedback,
    build_claim_snapshot,
    generate_feedback_summary,
    compute_calibration_stats,
    load_feedback_log,
)


CLAIMS_HISTORY_FILE = Path(__file__).parent / "claims_history.json"


def load_recent_claims(limit: int = 10) -> list:
    """Load recent claims from history."""
    if not CLAIMS_HISTORY_FILE.exists():
        return []
    try:
        history = json.loads(CLAIMS_HISTORY_FILE.read_text(encoding="utf-8"))
        return history[-limit:]
    except json.JSONDecodeError:
        return []


def list_claims() -> None:
    """Display recent claims for feedback."""
    claims = load_recent_claims(10)
    if not claims:
        print("No claims in history.")
        return
    
    print("\nRecent Claims (for feedback):\n")
    for i, claim in enumerate(claims, 1):
        insurer = claim.get("insurer_detected", "Unknown")
        decision = claim.get("parsed_decision", "Unknown")
        confidence = claim.get("confidence_score", 0.0)
        fraud = "[FRAUD]" if claim.get("fraud_detected") else "[VALID]"
        
        print(
            f"{i}. [{fraud}] {insurer} -> Decision: {decision} "
            f"(Confidence: {confidence})"
        )
    print()


def add_claim_feedback() -> None:
    """Interactively add feedback on a claim."""
    claims = load_recent_claims(10)
    if not claims:
        print("[ERROR] No claims to provide feedback on.")
        return
    
    list_claims()
    
    try:
        claim_idx = int(input("Enter claim number to provide feedback (or 0 to cancel): "))
        if claim_idx == 0:
            return
        if claim_idx < 1 or claim_idx > len(claims):
            print("[ERROR] Invalid claim number.")
            return
        
        claim = claims[claim_idx - 1]
        claim_id = f"claim_{claim_idx}"
        original_decision = claim.get("parsed_decision", "Unknown")
        original_confidence = claim.get("confidence_score", 0.5)
        
        print(f"\n[INFO] Providing feedback for: {original_decision} (Confidence: {original_confidence})")
        print("\nWhat should the CORRECT decision be?")
        print("1. Approved")
        print("2. Rejected")
        
        correct_choice = input("Enter choice (1 or 2): ").strip()
        if correct_choice not in {"1", "2"}:
            print("[ERROR] Invalid decision choice. Use 1 for Approved or 2 for Rejected.")
            return
        actual_decision = "Approved" if correct_choice == "1" else "Rejected"
        
        print("\n[?] Why is the AI decision incorrect? Select reason:")
        reasons = [
            "fraud_detected",
            "wrong_criteria_applied",
            "documentation_missing",
            "policy_mismatch",
            "waiting_period_not_met",
            "exclusion_applies",
            "amount_exceeds_limit",
            "other",
        ]
        for i, reason in enumerate(reasons, 1):
            print(f"{i}. {reason}")
        
        reason_choice = input("Enter reason number: ").strip()
        try:
            reason_idx = int(reason_choice) - 1
            if 0 <= reason_idx < len(reasons):
                feedback_reason = reasons[reason_idx]
            else:
                feedback_reason = "other"
        except ValueError:
            feedback_reason = "other"
        
        feedback_details = input("Any additional details? (optional, press Enter to skip): ").strip()
        
        # Record feedback
        feedback = add_feedback(
            claim_id=claim_id,
            original_decision=original_decision,
            original_confidence=original_confidence,
            actual_decision=actual_decision,
            feedback_reason=feedback_reason,
            feedback_details=feedback_details,
            claim_snapshot=build_claim_snapshot(claim),
        )
        
        is_error = feedback.get("is_false_positive") or feedback.get("is_false_negative")
        status = "[ERROR]" if is_error else "[OK]"
        
        print(f"\n[OK] Feedback recorded! {status}")
        print(f"   • Original: {original_decision} ({original_confidence})")
        print(f"   • Correct: {actual_decision}")
        print(f"   • Reason: {feedback_reason}")
        
    except KeyboardInterrupt:
        print("\n[CANCEL] Cancelled.")


def show_calibration() -> None:
    """Display model calibration report."""
    print(generate_feedback_summary())


def show_stats() -> None:
    """Show calibration statistics."""
    stats = compute_calibration_stats()
    print(json.dumps(stats, indent=2))


def show_recent_feedback(limit: int = 5) -> None:
    """Show recent feedback entries."""
    log = load_feedback_log()
    if not log:
        print("[ERROR] No feedback recorded yet.")
        return
    
    recent = log[-limit:]
    print(f"\n[INFO] Recent Feedback (last {len(recent)}):\n")
    for i, f in enumerate(recent, 1):
        status = "[ERROR]" if f.get("is_false_positive") or f.get("is_false_negative") else "[OK]"
        print(
            f"{i}. {status} [{f['original_decision']}→{f['actual_decision']}] "
            f"Conf:{f['original_confidence']} | Reason: {f['feedback_reason']}"
        )
        if f.get("feedback_details"):
            print(f"   Detail: {f['feedback_details']}")


def main():
    """Main feedback CLI menu."""
    while True:
        print("\n" + "="*60)
        print("           AI INSURANCE CLAIM FEEDBACK SYSTEM")
        print("="*60)
        print("\n1. Add feedback on a claim")
        print("2. View recent feedback")
        print("3. Show model calibration report")
        print("4. Show calibration statistics (JSON)")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            add_claim_feedback()
        elif choice == "2":
            show_recent_feedback()
        elif choice == "3":
            show_calibration()
        elif choice == "4":
            show_stats()
        elif choice == "5":
            print("[OK] Goodbye!")
            break
        else:
            print("[ERROR] Invalid choice.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "add":
            add_claim_feedback()
        elif cmd == "show":
            show_calibration()
        elif cmd == "stats":
            show_stats()
        elif cmd == "recent":
            limit = 10
            if len(sys.argv) > 2:
                try:
                    limit = max(1, int(sys.argv[2]))
                except ValueError:
                    print("[ERROR] recent expects an integer count. Using default 10.")
            show_recent_feedback(limit)
    else:
        main()
