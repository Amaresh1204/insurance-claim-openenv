---
title: Insurance Claim OpenEnv
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# Insurance Claim OpenEnv

**A real-world OpenEnv environment for training and evaluating AI agents on insurance claim adjudication.**

- **HF Space (live demo):** https://huggingface.co/spaces/AmareshKambhampati/insurance-claim-openenv
- **GitHub:** https://github.com/Amaresh1204/insurance-claim-openenv
- **OpenEnv interface:** `POST /step` · `POST /reset` · `GET /state` · `GET /tasks`

---

## Environment Description

Insurance claim adjudication is a high-stakes, real-world decision task that humans perform daily:
review a submitted PDF claim, detect fraud, check policy alignment, and approve or reject.

This OpenEnv environment wraps that full pipeline. An AI agent interacts via standard `step/reset/state`
HTTP calls and must:

1. Detect obvious insurer/policy-mismatch fraud and reject safely **(easy)**
2. Approve/reject based on policy criteria and data consistency **(medium)**
3. Produce calibrated confidence, grounded reasoning, and correct feedback override **(hard)**

Each task has a deterministic programmatic grader returning a reward in `[0.0, 1.0]` with six
partial-credit components, giving meaningful signal across the full trajectory.

---

## Quick Start

### Option A — Run against the live HF Space

```bash
pip install requests
ENV_BASE_URL=https://amareshkambhampati-insurance-claim-openenv.hf.space python demo.py
```

### Option B — Run locally with Docker

```bash
docker build -t insurance-claim-openenv .
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN insurance-claim-openenv
python demo.py
```

### Option C — Run locally without Docker

```bash
pip install -r requirements.txt
uvicorn openenv_api:app --host 0.0.0.0 --port 7860 &
python demo.py
```

### Run the reproducible baseline

```bash
python baseline_inference.py
```

### Run inference with a live LLM agent

```bash
export HF_TOKEN=<your-token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

---

## Repository Structure

```
├── openenv_api.py          # FastAPI service: /step /reset /state /tasks /health
├── openenv_core.py         # Typed models, task graders, reward breakdown, env class
├── openenv.yaml            # OpenEnv spec metadata
├── inference.py            # Hackathon inference script (OpenAI client, [START]/[STEP]/[END])
├── baseline_inference.py   # Reproducible baseline (mock/deterministic mode)
├── demo.py                 # End-to-end showcase: health → tasks → reset → steps → state
├── claim_pipeline.py       # Core claim processing pipeline
├── claim_assessor.py       # Structured extraction + policy fit + confidence
├── fraud_detector.py       # Multi-layer fraud detection
├── insurer_policy_loader.py# Insurer detection + criteria loading
├── feedback_handler.py     # Feedback recording + confidence calibration
├── app.py                  # Gradio web UI (Hugging Face Spaces frontend)
├── Dockerfile              # Container for HF Spaces (uvicorn on port 7860)
├── requirements.txt        # Python dependencies
├── sample.pdf              # Fraudulent claim sample (for easy task)
├── sample_valid.pdf        # Valid claim sample (for medium/hard tasks)
├── policy Criterias/       # Insurer-specific JSON policy rules
│   ├── hdfc_ergo_criteria.json
│   └── star_health_criteria.json
└── tests/
    └── test_openenv_core.py
```

---

## Action Space

Actions are sent as a `StepAction` JSON body to `POST /step`:

| Field | Type | Description |
|---|---|---|
| `pdf_path` | `str` | Path to the claim PDF (`sample.pdf` or `sample_valid.pdf`) |
| `task_id` | `str` | One of: `easy_fraud_detection`, `medium_policy_alignment`, `hard_calibrated_reasoning` |
| `use_mock_model` | `bool` | `true` for deterministic/reproducible mode; `false` for live LLM |
| `submit_feedback` | `bool` | `true` to submit a correction on the previous observation |
| `correct_decision` | `str?` | `"Approved"` or `"Rejected"` — required when `submit_feedback=true` |
| `feedback_reason` | `str` | `Policy mismatch` · `Fraud misclassification` · `Missing data` · `Incorrect validation` · `Other` |
| `feedback_details` | `str` | Optional free-text explanation |

---

## Observation Space

`StepResult` contains a typed `Observation`:

| Field | Type | Description |
|---|---|---|
| `parsed_decision` | `str` | `"Approved"` or `"Rejected"` |
| `confidence_score` | `float` | `[0.0, 1.0]` — multi-factor calibrated confidence |
| `fraud_detected` | `bool` | Whether fraud was flagged |
| `insurer_detected` | `str` | Detected insurer name |
| `policy_fit_score` | `float` | `[0.0, 1.0]` — weighted policy alignment score |
| `recommendation` | `str` | Human-readable recommendation text |
| `reward_model` | `float` | Raw LLM reward signal |
| `structured_claim_data` | `dict` | 15+ extracted fields (dates, amounts, policy no., etc.) |

The full response also contains `reward`, `score`, `done`, `reward_breakdown`, and `info`.

---

## Task Descriptions

### Task 1 — `easy_fraud_detection` (Easy)

**Objective:** Detect insurer/policy prefix mismatch in `sample.pdf` and reject the claim.

`sample.pdf` contains a claim with mismatched insurer and policy identifiers. The agent must output
`Rejected` with a reason mentioning fraud/mismatch/validation.

**Grader weights:**
- Decision correctness (Rejected): 35%
- Fraud detection flag: 35%
- Confidence ≥ 0.8: 10%
- Fraud-related explanation keywords: 20%

---

### Task 2 — `medium_policy_alignment` (Medium)

**Objective:** Approve `sample_valid.pdf` by correctly aligning with HDFC ERGO policy criteria.

The agent must output `Approved` (no fraud, policy fit high) with a reason referencing policy
coverage, hospitalization, or criteria terms.

**Grader weights:**
- Decision correctness (Approved): 30%
- No fraud flag: 20%
- Policy fit score: 30%
- Confidence ≥ 0.7: 10%
- Policy-grounded explanation: 10%

---

### Task 3 — `hard_calibrated_reasoning` (Hard)

**Objective:** Produce a valid decision, calibrated confidence, grounded ≥8-word reasoning,
a non-empty `final_recommendation`, and correct feedback override via `submit_feedback=true`.

All six reward components are active. The agent must both make the right call *and* demonstrate
it can self-correct via the feedback mechanism.

**Grader weights:**
- Decision validity: 20%
- Fraud consistency rule: 20%
- Policy fit score: 25%
- Confidence calibration: 15%
- Explanation quality (length + grounding + recommendation): 15%
- Feedback override used: 5%

---

## Reward Function

Each task grader returns a `score ∈ [0.0, 1.0]` and a full `RewardBreakdown`:

```json
{
  "decision_component":     0.0–1.0,
  "fraud_component":        0.0–1.0,
  "policy_component":       0.0–1.0,
  "confidence_component":   0.0–1.0,
  "explanation_component":  0.0–1.0,
  "feedback_component":     0.0–1.0
}
```

This gives **partial credit** across the trajectory — the agent receives a meaningful signal
even for partially correct decisions. The weights differ by task difficulty, requiring more
nuanced behaviour at higher levels.

---

## Baseline Scores

Produced by `baseline_inference.py` with `use_mock_model=true` (fully deterministic):

| Task | Difficulty | Score |
|---|---|---|
| `easy_fraud_detection` | Easy | 1.00 |
| `medium_policy_alignment` | Medium | 1.00 |
| `hard_calibrated_reasoning` | Hard | 0.97 |

Run to reproduce:
```bash
python baseline_inference.py
```

---

## OpenEnv Interface

All endpoints are live on the HF Space and locally via Docker:

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode, clear history |
| `/step` | POST | Run one task step, get observation + reward |
| `/state` | GET | Current episode state and accumulated scores |
| `/tasks` | GET | List all tasks with difficulty + description |
| `/health` | GET | Liveness check |

`openenv.yaml` declares all models and task metadata. Passes `openenv validate`.

---

## Episode Boundaries

- `max_steps = 3` (one per task)
- `done = true` after 3 steps
- Stepping past `max_steps` raises a `400` error
- `reset()` clears `claims_history.json`, `feedback_log.json`, `memory.json`

---

## Setup & Installation

```bash
# Clone
git clone https://github.com/Amaresh1204/insurance-claim-openenv.git
cd insurance-claim-openenv

# Install
pip install -r requirements.txt

# Environment variables
export HF_TOKEN=<your-huggingface-token>
export API_BASE_URL=https://router.huggingface.co/v1      # or your endpoint
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct               # or your model

# Start server
uvicorn openenv_api:app --host 0.0.0.0 --port 7860

# Run demo (new terminal)
python demo.py

# Run LLM inference agent
python inference.py

# Run deterministic baseline
python baseline_inference.py

# Run tests
python -m pytest tests/
```

---

## Docker

```bash
docker build -t insurance-claim-openenv .
docker run -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  insurance-claim-openenv
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (inference) | — | Hugging Face API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` | No | `http://localhost:7860` | OpenEnv server URL (for demo/inference) |

---

## Why Insurance Claim Adjudication?

Insurance adjudication is a genuine high-value task:
- Processes billions of dollars annually
- Requires multi-step reasoning (fraud check → policy check → decision)
- Has clear, programmable ground truth (fraud = reject, policy fit = approve)
- Benefits directly from AI agent improvement (faster, more consistent decisions)
- Difficulty scales naturally: easy fraud is obvious; hard cases require calibrated reasoning

This fills a real gap in the OpenEnv ecosystem — no prior insurance adjudication environment existed.
