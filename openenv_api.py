"""FastAPI wrapper exposing OpenEnv step/reset/state APIs and root UI."""

from fastapi import FastAPI, HTTPException
import gradio as gr

from openenv_core import InsuranceClaimOpenEnv, StepAction, StepResult, ResetRequest, ResetResult, StateResult, TASKS
import app as gradio_ui



env = InsuranceClaimOpenEnv()
app = FastAPI(
    title="Insurance Claim OpenEnv",
    version="1.0.0",
    description=(
        "Real-world claim adjudication environment with task graders, "
        "partial rewards, and OpenEnv-style step/reset/state APIs."
    ),
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": TASKS}


@app.post("/reset", response_model=ResetResult)
def reset_api(request: ResetRequest | None = None) -> ResetResult:
    try:
        clear_json = True if request is None else request.clear_json
        return env.reset(clear_json=clear_json)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state", response_model=StateResult)
def state_api() -> StateResult:
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/step",
    response_model=StepResult,
    summary="Run one environment step",
    description=(
        "Run one task step and return typed observation, reward, done, and info.\n\n"
        "Feedback workflow:\n"
        "1. Call `/step` with `submit_feedback=false` to create the initial observation.\n"
        "2. Call `/step` again with `submit_feedback=true` to submit correction feedback.\n"
        "3. When `submit_feedback=true`, set `correct_decision` to `Approved` or `Rejected`.\n\n"
        "Recommended `feedback_reason` values:\n"
        "- `Policy mismatch`\n"
        "- `Fraud misclassification`\n"
        "- `Missing data`\n"
        "- `Incorrect validation`\n"
        "- `Other`\n\n"
        "`feedback_details` is optional free-text context for graders and analysis."
    ),
)
def step_api(action: StepAction) -> StepResult:
    # Feedback usage on /step:
    # 1) Run a normal step first with submit_feedback=false.
    # 2) Submit feedback in a later step with submit_feedback=true.
    # 3) When submit_feedback=true, correct_decision must be Approved or Rejected.
    # 4) feedback_reason is a category (Policy mismatch, Fraud misclassification,
    #    Missing data, Incorrect validation, Other), and feedback_details is optional text.
    try:
        return env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Serve Gradio UI at root URL while keeping API endpoints (e.g. /docs, /step) available.
app = gr.mount_gradio_app(app, gradio_ui.demo, path="/")
