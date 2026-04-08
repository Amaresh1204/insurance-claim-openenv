"""Core OpenEnv runtime for insurance claim adjudication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from feedback_handler import add_feedback, build_claim_snapshot
from claim_pipeline import run_inference


ROOT_DIR = Path(__file__).parent
CLAIMS_HISTORY_FILE = ROOT_DIR / "claims_history.json"
FEEDBACK_LOG_FILE = ROOT_DIR / "feedback_log.json"
MEMORY_FILE = ROOT_DIR / "memory.json"


class StepAction(BaseModel):
	"""Action sent by an agent for one environment step."""

	pdf_path: str = Field(..., description="Path to claim PDF.")
	task_id: str = Field("easy_fraud_detection", description="Task grader to apply.")
	use_mock_model: bool = Field(
		True,
		description="Use deterministic policy-based model path for reproducible scores.",
	)
	submit_feedback: bool = Field(
		False,
		description=(
			"If true, submit feedback for the previous observation. "
			"Call one normal step first, then send feedback on the next step."
		),
	)
	correct_decision: Optional[str] = Field(
		None,
		description=(
			"Approved or Rejected. Required when submit_feedback=true."
		),
	)
	feedback_reason: str = Field(
		"Policy mismatch",
		description=(
			"Feedback category. Suggested values: Policy mismatch, Fraud misclassification, "
			"Missing data, Incorrect validation, Other."
		),
	)
	feedback_details: str = Field(
		"",
		description="Optional free-text details explaining why the decision was incorrect.",
	)


class Observation(BaseModel):
	"""Typed observation payload returned to the agent."""

	parsed_decision: str
	confidence_score: float
	fraud_detected: bool
	insurer_detected: str
	policy_fit_score: float
	recommendation: str
	reward_model: float
	structured_claim_data: Dict = Field(default_factory=dict)


class RewardBreakdown(BaseModel):
	"""Task-grade breakdown so agents get partial learning signals."""

	decision_component: float
	fraud_component: float
	policy_component: float
	confidence_component: float
	explanation_component: float
	feedback_component: float


class StepResult(BaseModel):
	"""Typed output of one step() call."""

	episode_id: str
	step_count: int
	task_id: str
	reward: float
	done: bool
	score: float
	observation: Observation
	reward_breakdown: RewardBreakdown
	info: Dict = Field(default_factory=dict)


class ResetRequest(BaseModel):
	"""Typed input for reset()."""

	clear_json: bool = Field(
		True,
		description="When true, reset clears claims_history.json, feedback_log.json and memory.json.",
	)


class ResetResult(BaseModel):
	"""Typed output of reset()."""

	episode_id: str
	cleared_files: List[str]
	message: str


class StateResult(BaseModel):
	"""Typed output of state()."""

	episode_id: str
	step_count: int
	total_reward: float
	last_task_id: Optional[str]
	last_observation: Optional[Observation]
	task_scores: Dict[str, float]
	max_steps: int
	episode_done: bool


TASKS = {
	"easy_fraud_detection": {
		"difficulty": "easy",
		"description": "Detect obvious insurer/policy mismatch fraud and reject safely.",
	},
	"medium_policy_alignment": {
		"difficulty": "medium",
		"description": "Approve/reject based on policy-fit checks and insurer criteria.",
	},
	"hard_calibrated_reasoning": {
		"difficulty": "hard",
		"description": "Produce consistent decision, calibrated confidence, and clear reasoning.",
	},
}


def _clamp01(value: float) -> float:
	return max(0.0, min(1.0, round(value, 4)))


def _safe_reason_text(result: dict) -> str:
	reason = (result.get("parsed_reason") or "").strip().lower()
	if not reason:
		reason = (result.get("decision_output") or "").strip().lower()
	return reason


def _grade_easy(result: dict) -> Tuple[float, RewardBreakdown]:
	decision = result.get("parsed_decision") == "Rejected"
	fraud = bool(result.get("fraud_detected"))
	reason = _safe_reason_text(result)
	reason_ok = any(k in reason for k in ["fraud", "mismatch", "validation failed"])

	breakdown = RewardBreakdown(
		decision_component=1.0 if decision else 0.0,
		fraud_component=1.0 if fraud else 0.0,
		policy_component=1.0 if result.get("policy_fit", {}).get("score", 0.0) >= 0.0 else 0.0,
		confidence_component=1.0 if result.get("confidence_score", 0.0) >= 0.8 else 0.4,
		explanation_component=1.0 if reason_ok else 0.2,
		feedback_component=0.0,
	)

	score = _clamp01(
		(0.35 * breakdown.decision_component)
		+ (0.35 * breakdown.fraud_component)
		+ (0.10 * breakdown.confidence_component)
		+ (0.20 * breakdown.explanation_component)
	)
	return score, breakdown


def _grade_medium(result: dict) -> Tuple[float, RewardBreakdown]:
	decision = result.get("parsed_decision") == "Approved"
	fraud = not bool(result.get("fraud_detected"))
	policy_fit = result.get("policy_fit", {}).get("score", 0.0)
	conf = result.get("confidence_score", 0.0)
	reason = _safe_reason_text(result)
	reason_ok = any(k in reason for k in ["cover", "policy", "hospitalization", "criteria"])

	breakdown = RewardBreakdown(
		decision_component=1.0 if decision else 0.0,
		fraud_component=1.0 if fraud else 0.0,
		policy_component=_clamp01(policy_fit),
		confidence_component=1.0 if conf >= 0.7 else _clamp01(conf),
		explanation_component=1.0 if reason_ok else 0.2,
		feedback_component=0.0,
	)

	score = _clamp01(
		(0.30 * breakdown.decision_component)
		+ (0.20 * breakdown.fraud_component)
		+ (0.30 * breakdown.policy_component)
		+ (0.10 * breakdown.confidence_component)
		+ (0.10 * breakdown.explanation_component)
	)
	return score, breakdown


def _grade_hard(result: dict) -> Tuple[float, RewardBreakdown]:
	decision = result.get("parsed_decision") in {"Approved", "Rejected"}
	fraud_rule = not (result.get("fraud_detected") and result.get("parsed_decision") == "Approved")
	policy_fit = _clamp01(result.get("policy_fit", {}).get("score", 0.0))
	conf = result.get("confidence_score", 0.0)
	reason = _safe_reason_text(result)
	reason_len_ok = len(reason.split()) >= 8
	reason_grounded = any(k in reason for k in ["policy", "coverage", "exclusion", "hospitalization", "fraud"])
	recommendation_ok = bool(result.get("final_recommendation"))
	feedback_used = 1.0 if result.get("feedback_override_applied") else 0.5

	explanation_score = 1.0 if (reason_len_ok and reason_grounded and recommendation_ok) else 0.3

	breakdown = RewardBreakdown(
		decision_component=1.0 if decision else 0.0,
		fraud_component=1.0 if fraud_rule else 0.0,
		policy_component=policy_fit,
		confidence_component=_clamp01(conf),
		explanation_component=explanation_score,
		feedback_component=feedback_used,
	)

	score = _clamp01(
		(0.20 * breakdown.decision_component)
		+ (0.20 * breakdown.fraud_component)
		+ (0.25 * breakdown.policy_component)
		+ (0.15 * breakdown.confidence_component)
		+ (0.15 * breakdown.explanation_component)
		+ (0.05 * breakdown.feedback_component)
	)
	return score, breakdown


def grade_task(task_id: str, result: dict) -> Tuple[float, RewardBreakdown]:
	"""Task-specific grader with partial credit signals in [0.0, 1.0]."""
	if task_id == "easy_fraud_detection":
		return _grade_easy(result)
	if task_id == "medium_policy_alignment":
		return _grade_medium(result)
	return _grade_hard(result)


def build_observation(result: dict) -> Observation:
	"""Normalize inference result into typed observation model."""
	return Observation(
		parsed_decision=result.get("parsed_decision", "Unknown"),
		confidence_score=float(result.get("confidence_score", 0.0)),
		fraud_detected=bool(result.get("fraud_detected", False)),
		insurer_detected=result.get("insurer_detected", "Unknown"),
		policy_fit_score=float(result.get("policy_fit", {}).get("score", 0.0)),
		recommendation=result.get("final_recommendation", "Needs manual review."),
		reward_model=float(result.get("reward", 0.0)),
		structured_claim_data=result.get("structured_claim_data", {}) or {},
	)


@dataclass
class _EpisodeState:
	episode_id: str
	step_count: int
	total_reward: float
	last_task_id: Optional[str]
	last_observation: Optional[Observation]
	last_result_raw: Optional[dict]
	task_scores: Dict[str, float]


class InsuranceClaimOpenEnv:
	"""OpenEnv-compatible environment for claim decision workflows."""

	def __init__(self, max_steps: int = 3):
		self.max_steps = max(1, int(max_steps))
		self._state = _EpisodeState(
			episode_id=str(uuid4()),
			step_count=0,
			total_reward=0.0,
			last_task_id=None,
			last_observation=None,
			last_result_raw=None,
			task_scores={},
		)

	def reset(self, clear_json: bool = True) -> ResetResult:
		"""Reset episode state and optionally clear persisted JSON history."""
		cleared = []
		if clear_json:
			for path, default_payload in [
				(CLAIMS_HISTORY_FILE, []),
				(FEEDBACK_LOG_FILE, []),
				(MEMORY_FILE, []),
			]:
				path.write_text(json.dumps(default_payload, indent=2), encoding="utf-8")
				cleared.append(path.name)

		self._state = _EpisodeState(
			episode_id=str(uuid4()),
			step_count=0,
			total_reward=0.0,
			last_task_id=None,
			last_observation=None,
			last_result_raw=None,
			task_scores={},
		)
		return ResetResult(
			episode_id=self._state.episode_id,
			cleared_files=cleared,
			message="Environment reset complete.",
		)

	def state(self) -> StateResult:
		"""Return current episode state."""
		episode_done = self._state.step_count >= self.max_steps
		return StateResult(
			episode_id=self._state.episode_id,
			step_count=self._state.step_count,
			total_reward=round(self._state.total_reward, 4),
			last_task_id=self._state.last_task_id,
			last_observation=self._state.last_observation,
			task_scores=self._state.task_scores,
			max_steps=self.max_steps,
			episode_done=episode_done,
		)

	def step(self, action: StepAction) -> StepResult:
		"""Run one environment step with task grading and partial reward signals."""
		if self._state.step_count >= self.max_steps:
			raise ValueError("Episode is complete. Call reset() before taking more steps.")

		if action.task_id not in TASKS:
			raise ValueError(f"Unknown task_id '{action.task_id}'. Available: {list(TASKS)}")

		if action.submit_feedback:
			if not self._state.last_result_raw:
				raise ValueError("No previous observation available for feedback submission.")
			if action.correct_decision not in {"Approved", "Rejected"}:
				raise ValueError("correct_decision must be 'Approved' or 'Rejected' when submit_feedback=true.")

			previous = self._state.last_result_raw
			add_feedback(
				claim_id=f"episode_{self._state.episode_id}_step_{self._state.step_count}",
				original_decision=previous.get("parsed_decision", "Unknown"),
				original_confidence=previous.get("confidence_score", 0.5),
				actual_decision=action.correct_decision,
				feedback_reason=action.feedback_reason,
				feedback_details=action.feedback_details,
				claim_snapshot=build_claim_snapshot(previous),
			)

		result = run_inference(action.pdf_path, use_mock_model=action.use_mock_model)
		if "error" in result:
			raise ValueError(result["error"])

		score, breakdown = grade_task(action.task_id, result)
		observation = build_observation(result)

		self._state.step_count += 1
		self._state.total_reward += score
		self._state.last_task_id = action.task_id
		self._state.last_observation = observation
		self._state.last_result_raw = result
		self._state.task_scores[action.task_id] = score
		done = self._state.step_count >= self.max_steps

		return StepResult(
			episode_id=self._state.episode_id,
			step_count=self._state.step_count,
			task_id=action.task_id,
			reward=score,
			done=done,
			score=score,
			observation=observation,
			reward_breakdown=breakdown,
			info={
				"difficulty": TASKS[action.task_id]["difficulty"],
				"task_description": TASKS[action.task_id]["description"],
				"model_reward": result.get("reward", 0.0),
				"mock_mode": action.use_mock_model,
				"remaining_steps": max(self.max_steps - self._state.step_count, 0),
			},
		)
