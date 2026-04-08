"""Reproducible baseline runner for the OpenEnv insurance environment.
Emits stdout strictly according to the hackathon [START] [STEP] [END] standards.
"""

from typing import List
from openenv_core import InsuranceClaimOpenEnv, StepAction

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = "null") -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_baseline() -> None:
    env = InsuranceClaimOpenEnv()
    env.reset(clear_json=True)

    actions = [
        StepAction(
            pdf_path="sample.pdf",
            task_id="easy_fraud_detection",
            use_mock_model=True,
        ),
        StepAction(
            pdf_path="sample_valid.pdf",
            task_id="medium_policy_alignment",
            use_mock_model=True,
        ),
        StepAction(
            pdf_path="sample_valid.pdf",
            task_id="hard_calibrated_reasoning",
            use_mock_model=True,
        ),
    ]

    for action in actions:
        log_start(task=action.task_id, env="insurance_claim_openenv", model="mock_baseline")
        
        result = env.step(action)
        reward = result.score
        
        log_step(step=1, action=action.model_dump_json(), reward=reward, done=True)
        log_end(success=(reward >= 0.5), steps=1, score=reward, rewards=[reward])

if __name__ == "__main__":
    run_baseline()
