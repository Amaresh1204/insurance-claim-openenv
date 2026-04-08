"""FastAPI wrapper exposing OpenEnv step/reset/state APIs."""

from fastapi import FastAPI, HTTPException

from openenv_core import InsuranceClaimOpenEnv, StepAction, StepResult, ResetRequest, ResetResult, StateResult, TASKS



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
def reset_api(request: ResetRequest) -> ResetResult:
    try:
        return env.reset(clear_json=request.clear_json)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state", response_model=StateResult)
def state_api() -> StateResult:
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult)
def step_api(action: StepAction) -> StepResult:
    try:
        return env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
