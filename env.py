"""OpenEnv-style environment for insurance claim evaluation."""

import re


class InsuranceEnv:
    """Reinforcement-learning-style environment that scores claim decisions."""

    def __init__(self):
        self.state = None

    def reset(self, document_text: str) -> dict:
        """Reset the environment with a new document.

        Args:
            document_text: Raw text extracted from the insurance PDF.

        Returns:
            The initial observation/state dict.
        """
        self.state = {"document": document_text}
        return self.state

    def step(self, action: str) -> tuple:
        """Evaluate the LLM's decision and return a reward.

        Reward logic:
            - 1.0  if the decision is 'Approved' and reasoning mentions coverage.
            - 0.8  if the decision is 'Rejected' (valid conservative call).
            - 0.3  otherwise (unclear or malformed output).

        Args:
            action: The raw text output from the LLM.

        Returns:
            Tuple of (state, reward, done, info).
        """
        action_text = (action or "").strip()
        action_lower = action_text.lower()

        decision_match = re.search(r"decision\s*:\s*(approved|rejected)", action_lower)
        reason_match = re.search(r"reason\s*:\s*(.+)", action_lower, flags=re.DOTALL)

        decision = decision_match.group(1) if decision_match else ""
        reason = reason_match.group(1).strip() if reason_match else ""

        coverage_keywords = (
            "cover",
            "covered",
            "coverage",
            "hospitalization",
            "surgery",
            "surgeries",
            "policy covers",
        )
        has_coverage_reason = any(keyword in reason for keyword in coverage_keywords)

        if decision == "approved":
            reward = 1.0 if has_coverage_reason else 0.6
        elif decision == "rejected":
            reward = 0.8 if reason else 0.6
        else:
            # Fallback for malformed responses that do not follow required format.
            reward = 0.3

        done = True
        return self.state, reward, done, {}
