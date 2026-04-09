"""
Inference Script — Insurance Claim OpenEnv
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# OpenEnv environment URL (local Docker or HF Space)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "insurance-claim-openenv"
MAX_STEPS = 2  # per task: step 1 = process claim, step 2 = submit feedback
TEMPERATURE = 0.2
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5

# Task definitions — easy -> medium -> hard
TASKS = [
    {
        "task_id": "easy_fraud_detection",
        "pdf_path": "sample.pdf",
        "description": "Detect insurer/policy mismatch fraud and reject with clear rationale.",
    },
    {
        "task_id": "medium_policy_alignment",
        "pdf_path": "sample_valid.pdf",
        "description": "Align decision with policy checks and claim data consistency.",
    },
    {
        "task_id": "hard_calibrated_reasoning",
        "pdf_path": "sample_valid.pdf",
        "description": "Produce calibrated confidence and grounded reasoning.",
    },
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert insurance claim adjudication agent.
    You interact with an Insurance Claim OpenEnv environment via step/reset/state APIs.

    Given an observation from the environment containing:
    - parsed_decision, confidence_score, fraud_detected, insurer_detected
    - policy_fit_score, recommendation, structured_claim_data

    Your job is to analyze the observation and decide the next action.
    For each task you must evaluate the claim and provide feedback if the AI decision was wrong.

    Respond ONLY with a valid JSON object (no markdown fences) with these fields:
    {
        "analysis": "Brief analysis of the observation",
        "should_submit_feedback": true/false,
        "correct_decision": "Approved" or "Rejected",
        "feedback_reason": "one of: Policy mismatch, Fraud misclassification, Missing data, Incorrect validation, Other",
        "feedback_details": "optional details"
    }
""").strip()


# ---------------------------------------------------------------------------
# Structured stdout logging (mandatory format)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------
def env_reset() -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"clear_json": True},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def env_state() -> dict:
    resp = requests.get(f"{ENV_BASE_URL}/state", timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------
def build_user_prompt(task: dict, observation: dict, step_num: int) -> str:
    obs_summary = json.dumps(observation, indent=2, default=str)
    return textwrap.dedent(f"""
        Task: {task['description']}
        Task ID: {task['task_id']}
        Step: {step_num}

        The environment processed the claim PDF "{task['pdf_path']}" and returned
        the following observation:

        {obs_summary}

        Based on this observation:
        1. Analyze whether the AI's decision (parsed_decision) is correct.
        2. If fraud_detected is true, the claim should likely be Rejected.
        3. Check if policy_fit_score and confidence_score are reasonable.
        4. Decide if you should submit feedback to correct the AI.

        Respond with a JSON object as instructed.
    """).strip()


def parse_agent_response(response_text: str) -> dict:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "analysis": "Could not parse response",
            "should_submit_feedback": False,
            "correct_decision": "Approved",
            "feedback_reason": "Other",
            "feedback_details": "",
        }


def get_agent_response(client: OpenAI, task: dict, observation: dict, step_num: int) -> dict:
    user_prompt = build_user_prompt(task, observation, step_num)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        return parse_agent_response(response_text)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {
            "analysis": f"LLM error: {exc}",
            "should_submit_feedback": False,
            "correct_decision": "Approved",
            "feedback_reason": "Other",
            "feedback_details": "",
        }


# ---------------------------------------------------------------------------
# Run a single task (separate episode per task)
# ---------------------------------------------------------------------------
def run_task(client: OpenAI, task: dict) -> float:
    """Run one task as a full episode. Returns the task score in [0, 1]."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task["task_id"], env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment for this task
        env_reset()

        # --- Step 1: Process the claim ---
        action_step1 = {
            "pdf_path": task["pdf_path"],
            "task_id": task["task_id"],
            "use_mock_model": False,
            "submit_feedback": False,
        }

        result = env_step(action_step1)
        observation = result.get("observation", {})
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        last_error = result.get("info", {}).get("error")

        rewards.append(reward)
        steps_taken = 1
        score = reward

        action_desc = f"process_claim({task['pdf_path']},{task['task_id']})"
        log_step(step=1, action=action_desc, reward=reward, done=done, error=last_error)

        if not done:
            # --- Step 2: LLM analyzes observation and optionally submits feedback ---
            agent_action = get_agent_response(client, task, observation, step_num=2)

            action_step2 = {
                "pdf_path": task["pdf_path"],
                "task_id": task["task_id"],
                "use_mock_model": False,
                "submit_feedback": agent_action.get("should_submit_feedback", False),
                "correct_decision": agent_action.get("correct_decision", "Approved"),
                "feedback_reason": agent_action.get("feedback_reason", "Other"),
                "feedback_details": agent_action.get("feedback_details", ""),
            }

            result2 = env_step(action_step2)
            reward2 = result2.get("reward", 0.0)
            done2 = result2.get("done", True)
            last_error = result2.get("info", {}).get("error")

            rewards.append(reward2)
            steps_taken = 2

            feedback_flag = "feedback" if action_step2["submit_feedback"] else "no_feedback"
            action_desc2 = f"analyze_and_{feedback_flag}({task['task_id']})"
            log_step(step=2, action=action_desc2, reward=reward2, done=done2, error=last_error)

            # Score is the last reward (the task grader score for the final step)
            score = reward2

        # Clamp score to strictly (0, 1) — validator rejects exact 0.0 and 1.0
        score = max(0.01, min(0.99, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        last_error = str(exc)
        print(f"[DEBUG] Task {task['task_id']} error: {exc}", flush=True)

    finally:
        # Ensure score is strictly in (0, 1) even on error paths
        score = max(0.01, min(0.99, score))
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not API_KEY:
        print("[DEBUG] ERROR: HF_TOKEN or API_KEY must be set.", flush=True)
        for task in TASKS:
            log_start(task=task["task_id"], env=BENCHMARK, model=MODEL_NAME or "unknown")
            log_end(success=False, steps=0, score=0.01, rewards=[])
        return

    if not MODEL_NAME:
        print("[DEBUG] ERROR: MODEL_NAME must be set.", flush=True)
        for task in TASKS:
            log_start(task=task["task_id"], env=BENCHMARK, model="unknown")
            log_end(success=False, steps=0, score=0.01, rewards=[])
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: List[float] = []
    for task in TASKS:
        task_score = run_task(client, task)
        all_scores.append(task_score)

    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[DEBUG] Mean score across {len(TASKS)} tasks: {mean_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
