"""Load and match insurer policy criteria from the policy folder."""

import json
from pathlib import Path


POLICY_CRITERIAS_DIR = Path(__file__).parent / "policy Criterias"

# Mapping of insurer names in PDFs to criteria file names
INSURER_MAPPING = {
    "hdfc": "hdfc_ergo_criteria.json",
    "hdfc ergo": "hdfc_ergo_criteria.json",
    "health suraksha": "hdfc_ergo_criteria.json",
    "star health": "star_health_criteria.json",
    "star": "star_health_criteria.json",
}


def detect_insurer_from_text(document_text: str) -> str:
    """Extract insurer/hospital name from document text.
    
    Returns insurer name or empty string if not found.
    """
    text_lower = document_text.lower()
    
    # Check for known insurers in text
    for insurer_key in INSURER_MAPPING.keys():
        if insurer_key in text_lower:
            return insurer_key
    
    return ""


def load_policy_criteria(insurer_name: str) -> dict:
    """Load policy criteria JSON for a given insurer.
    
    Args:
        insurer_name: Insurer key (e.g., 'hdfc', 'star health')
    
    Returns:
        Dict with policy criteria, or empty dict if not found.
    """
    insurer_name_lower = insurer_name.lower().strip()
    
    if insurer_name_lower not in INSURER_MAPPING:
        return {}
    
    criteria_file = POLICY_CRITERIAS_DIR / INSURER_MAPPING[insurer_name_lower]
    
    if not criteria_file.exists():
        return {}
    
    try:
        with open(criteria_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def format_criteria_for_prompt(criteria: dict) -> str:
    """Format policy criteria into a readable prompt section."""
    if not criteria:
        return ""
    
    insurer = criteria.get("insurer", "Unknown")
    policy_name = criteria.get("policy_name", "")
    approval_criteria = criteria.get("claim_approval_criteria", {})
    exclusions = criteria.get("typical_exclusions", [])
    
    lines = [
        f"\n--- {insurer} Policy Criteria ---",
        f"Policy: {policy_name}",
        "\nApproval Requirements:",
    ]
    
    # Add key criteria
    if approval_criteria.get("hospitalization_required"):
        min_hours = approval_criteria.get("minimum_hospitalization_hours", 24)
        lines.append(f"  • Minimum {min_hours} hours hospitalization required")
    
    if approval_criteria.get("waiting_period_completed"):
        wp = approval_criteria["waiting_period_completed"]
        lines.append(f"  • Waiting period: {wp.get('initial_waiting_days', 30)} days initial")
        lines.append(f"  • Pre-existing diseases: {wp.get('pre_existing_disease_years', 3)} years")
    
    docs = approval_criteria.get("documents_required", [])
    if docs:
        lines.append(f"  • Required documents: {', '.join(docs[:3])}...")
    
    if exclusions:
        lines.append("\nExclusions (auto-reject if applicable):")
        for excl in exclusions[:3]:
            lines.append(f"  ✗ {excl}")
    
    return "\n".join(lines)
