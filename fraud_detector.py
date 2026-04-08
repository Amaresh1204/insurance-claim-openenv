"""Fraud detection and document validation for insurance claims."""

import re


def extract_policy_fields(document_text: str) -> dict:
    """Extract key policy fields from document text.
    
    Returns:
        Dict with extracted fields: insurer, policy_number, hospital, etc.
    """
    text_lower = document_text.lower()
    fields = {}
    
    # Extract Insurer
    insurer_match = re.search(r"insurer\s*:\s*([^\n]+)", text_lower)
    if insurer_match:
        fields["insurer"] = insurer_match.group(1).strip()
    
    # Extract Policy Number (case-insensitive, more flexible)
    policy_match = re.search(r"policy\s*number\s*:\s*([a-z0-9\-]+)", document_text, re.IGNORECASE)
    if policy_match:
        fields["policy_number"] = policy_match.group(1).strip().upper()
    
    # Extract Hospital
    hospital_match = re.search(r"hospital\s*:\s*([^\n]+)", text_lower)
    if hospital_match:
        fields["hospital"] = hospital_match.group(1).strip()
    
    return fields


def validate_document_consistency(document_text: str) -> tuple:
    """Validate document for fraud indicators and inconsistencies.
    
    Returns:
        Tuple of (is_valid: bool, fraud_flags: list[str])
        fraud_flags contains descriptions of detected issues
    """
    fraud_flags = []
    fields = extract_policy_fields(document_text)
    
    insurer = fields.get("insurer", "").lower()
    policy_number = fields.get("policy_number", "").upper()
    
    # Check 1: Insurer vs Policy Number Prefix Mismatch
    if insurer and policy_number:
        # Extract prefix from policy number (first part before dash)
        prefix = policy_number.split("-")[0].lower() if "-" in policy_number else ""
        
        insurer_mapping = {
            "hdfc": ["hdfc", "health suraksha"],
            "star": ["star", "starhealth", "star health"],
            "bajaj": ["bajaj"],
            "aditya birla": ["aditya", "birla"],
        }
        
        # Find which insurer group the prefix belongs to
        prefix_group = None
        for group_key, aliases in insurer_mapping.items():
            if prefix in aliases or any(alias in prefix for alias in aliases):
                prefix_group = group_key
                break
        
        # Find which insurer group the insurer name belongs to
        insurer_group = None
        for group_key, aliases in insurer_mapping.items():
            if any(alias in insurer for alias in aliases):
                insurer_group = group_key
                break
        
        # Mismatch detected
        if prefix_group and insurer_group and prefix_group != insurer_group:
            fraud_flags.append(
                f"MISMATCH: Policy number prefix '{prefix}' suggests {prefix_group.upper()} "
                f"but Insurer field states '{insurer.upper()}' (likely fraudulent)"
            )
    
    # Check 2: Missing critical fields
    if not insurer:
        fraud_flags.append("MISSING: Insurer name not found in document")
    if not policy_number:
        fraud_flags.append("MISSING: Policy number not found in document")
    
    is_valid = len(fraud_flags) == 0
    return is_valid, fraud_flags
