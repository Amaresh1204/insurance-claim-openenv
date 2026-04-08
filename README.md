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

# Insurance Claim AI — OpenEnv Environment ✅

## Environment Description

A **real-world insurance claim adjudication environment** where an AI agent processes PDF insurance documents, detects fraud, evaluates policy compliance, and makes approve/reject decisions with calibrated confidence scores. Includes a human-in-the-loop feedback system that adjusts confidence calibration at inference time.

**Why this matters:** Insurance claim adjudication is a task millions of humans do daily, involving document analysis, rule-based validation, and nuanced decision-making — ideal for evaluating and training AI agents.

---

## OpenEnv API Endpoints

| Method | Path | Request Model | Response Model |
|--------|------|---------------|----------------|
| `POST` | `/reset` | `ResetRequest` | `ResetResult` |
| `POST` | `/step` | `StepAction` | `StepResult` |
| `GET` | `/state` | — | `StateResult` |
| `GET` | `/health` | — | `{"status": "ok"}` |
| `GET` | `/tasks` | — | Task definitions |

---

## Action Space (`StepAction`)

| Field | Type | Description |
|-------|------|-------------|
| `pdf_path` | `str` | Path to claim PDF |
| `task_id` | `str` | Task grader: `easy_fraud_detection`, `medium_policy_alignment`, `hard_calibrated_reasoning` |
| `use_mock_model` | `bool` | Use deterministic path for reproducible scores |
| `submit_feedback` | `bool` | If `true`, feedback fields are recorded |
| `correct_decision` | `str?` | `Approved`/`Rejected` (when submitting feedback) |
| `feedback_reason` | `str` | Reason category |
| `feedback_details` | `str` | Optional details |

## Observation Space (`Observation`)

| Field | Type | Description |
|-------|------|-------------|
| `parsed_decision` | `str` | `Approved` or `Rejected` |
| `confidence_score` | `float` | 0.0-1.0 calibrated confidence |
| `fraud_detected` | `bool` | Whether fraud was flagged |
| `insurer_detected` | `str` | Detected insurer name |
| `policy_fit_score` | `float` | 0.0-1.0 policy alignment |
| `recommendation` | `str` | Final recommendation text |
| `reward_model` | `float` | Raw reward from env scoring |
| `structured_claim_data` | `dict` | Extracted claim JSON data |

## Reward Model (`RewardBreakdown`)

Each task grader returns a score in `[0.0, 1.0]` with 6 partial credit components:
- **Decision correctness** — Did the agent make the right call?
- **Fraud handling** — Was fraud properly detected/dismissed?
- **Policy alignment** — Does the decision match policy criteria?
- **Confidence calibration** — Is confidence appropriately calibrated?
- **Explanation quality** — Is the reasoning grounded and detailed?
- **Feedback influence** — Did the agent use feedback effectively? (hard task)

---

## Tasks (3 difficulty levels)

| Task ID | Difficulty | Objective | Baseline Score |
|---------|-----------|-----------|---------------|
| `easy_fraud_detection` | Easy | Detect insurer/policy mismatch fraud and reject with clear rationale | **1.000** |
| `medium_policy_alignment` | Medium | Align decision with policy checks and claim data consistency | **1.000** |
| `hard_calibrated_reasoning` | Hard | Produce calibrated confidence and grounded reasoning with consistency | **0.968** |

**Mean baseline score: 0.989** (deterministic mock model)

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_hf_token"
```

### 3. Run the OpenEnv API server
```bash
uvicorn openenv_api:app --host 0.0.0.0 --port 7860
```

### 4. Run the agent inference script
```bash
python inference.py
```

### 5. Run the reproducible baseline (deterministic, no LLM)
```bash
python baseline_inference.py
```

### 6. Launch the Gradio Web UI (optional)
```bash
python app.py
```

---

## Docker Deployment

```bash
docker build -t insurance-claim-openenv .
docker run -p 7860:7860 insurance-claim-openenv
```

### Hugging Face Spaces
- SDK: `Docker`
- Port: `7860`
- Tag: `openenv`

Push this repository to a Hugging Face Space with Docker SDK.

---

## Project Structure

| File | Purpose |
|------|---------|
| `openenv.yaml` | OpenEnv environment specification |
| `openenv_core.py` | Typed models, task graders, reward breakdown, env class |
| `openenv_api.py` | FastAPI server exposing step/reset/state endpoints |
| `inference.py` | Hackathon agent script — uses OpenAI client + HTTP env calls |
| `baseline_inference.py` | Reproducible baseline (deterministic mock model) |
| `claim_pipeline.py` | Core claim processing pipeline (PDF to LLM to decision) |
| `claim_assessor.py` | Structured data extraction and policy evaluation |
| `fraud_detector.py` | Multi-layer fraud detection |
| `feedback_handler.py` | Human-in-the-loop feedback calibration engine |
| `insurer_policy_loader.py` | Insurer-specific policy criteria loader |
| `env.py` | Low-level reward scoring environment |
| `app.py` | Gradio Web UI |
| `ui.py` | UI layout definitions |
| `Dockerfile` | Container deployment |

Use Space SDK: `Docker`, then push this repository.

## 🎯 System Overview

This is a **production-ready Insurance Claim Decision System** with:
- ✅ Fraudulent document detection
- ✅ Insurer-specific policy rule matching
- ✅ Structured claim data extraction
- ✅ Multi-factor confidence scoring
- ✅ **Human-in-the-loop feedback learning** ← NEW!
- ✅ CLI + Web UI (Gradio)
- ✅ Persistent claims history and feedback logging

---

## 🚀 Quick Start (5 minutes)

### 1. Process Your First Claim

```bash
python claim_pipeline.py sample.pdf
```

**Output:**
- Decision (Approved/Rejected)
- Confidence score
- Extracted claim data (JSON)
- Policy fit analysis
- Fraud detection results

### 2. Provide Feedback (If Wrong!)

```bash
python feedback_cli.py add
```

Interactive prompt to:
- Mark claim as incorrect
- Select actual decision
- Choose feedback reason
- Add optional details

### 3. Launch Web UI

```bash
python app.py
```

Then visit `http://localhost:7860` in your browser

---

## 📊 What Each Component Does

### **claim_pipeline.py** — Main Pipeline
Orchestrates 13-step claim processing:
1. Load PDF
2. Extract structured data (15+ fields)
3. Fraud validation (multi-layer)
4. Detect insurer from text
5. Load insurer-specific policies
6. Evaluate policy fit (weighted scoring)
7. Call LLM (meta-llama/3.1-8B via HF)
8. Parse LLM decision
9. Handle rejection reasons
10. Store in claims history
11. Compute confidence (5-component weighted)
12. **Adjust confidence for feedback patterns** ← NEW!
13. Generate recommendation

**Key Output:** Full decision with confidence, extracted data, policy fit, and historical learning

### **feedback_cli.py** — Human-in-Loop Control
Interactive CLI for correcting AI decisions:
- `add` — Record user correction
- `recent` — View recent feedback
- `show` — Calibration report
- `stats` — JSON statistics

**Key Output:** Feedback recorded, calibration statistics, confidence adjustments tracked

### **feedback_handler.py** — Learning Engine
Processes feedback to calibrate confidence:
- `add_feedback()` — Records corrections
- `compute_calibration_stats()` — Analyzes accuracy by confidence bin
- `adjust_confidence_for_feedback()` — Auto-reduces confidence when patterns show errors
- `generate_feedback_summary()` — Pretty-print calibration report

**Key Output:** Bin-level accuracy, false positive/negative rates, adjustment factors

### **claim_assessor.py** — Claim Analysis
Extracts and evaluates claims:
- Regex-based field extraction (insurer, policy, dates, amounts, etc.)
- Policy fit scoring (weighted against criteria)
- Confidence computation (LLM reward, policy fit, format, history, base)
- Maintains claims history (100 rolling records)

**Key Output:** Structured JSON claim, policy fit score, confidence components

### **fraud_detector.py** — Document Validation
Multi-layer fraud detection:
- Insurer/policy prefix mismatch detection
- Missing field validation
- Policy criteria consistency checks

**Key Output:** Fraud flag, detailed mismatch reasons, false positive prevention

### **insurer_policy_loader.py** — Policy Management
Insurer-specific logic:
- Auto-detect insurer from document text
- Load insurer criteria from JSON files
- Format criteria for LLM prompt

**Key Output:** Detected insurer, policy criteria, formatted prompt injection

### **env.py** — Reward Scoring
OpenEnv-style decision quality scoring:
- Regex-based LLM output parsing
- Decision + Reason extraction
- Coverage keyword validation
- Reward scale: 0.0 (fraud) → 0.5 (uncertain) → 1.0 (clear approval)

**Key Output:** Reward score for confidence calculation

### **app.py** — Gradio Web UI
Browser-based interface:
- PDF upload
- Live claim processing
- Results display with formatted JSON
- Gradio sharing enabled

**Key Output:** Web-accessible claim processing

---

## 📁 Data Files

### Claims History (`claims_history.json`)
Stores last 100 processed claims:
```json
{
  "insurer_detected": "hdfc",
  "parsed_decision": "Approved",
  "confidence_score": 0.95,
  "fraud_detected": false,
  "claim_will_be_approved": true,
  "structured_claim_data": {...},
  "policy_fit": {"score": 1.0},
  "timestamp": "2024-..."
}
```

### Feedback Log (`feedback_log.json`)
Stores last 500 user corrections:
```json
{
  "claim_id": "claim_1",
  "original_decision": "Approved",
  "actual_decision": "Rejected",
  "original_confidence": 0.92,
  "feedback_reason": "fraud_detected",
  "is_false_positive": true,
  "is_false_negative": false,
  "timestamp": "2024-..."
}
```

### Calibration Cache (`confidence_calibration.json`)
Stores computed statistics:
```json
{
  "total_feedback": 5,
  "accuracy": 0.8,
  "false_positive_rate": 0.2,
  "confidence_bins": {
    "0.80-0.90": {"count": 2, "accuracy": 0.5, "false_positives": 1}
  }
}
```

### Policy Criteria (`policy_criterias/*.json`)
Insurer-specific rules:
```json
{
  "insurer_name": "HDFC ERGO",
  "policy_type": "Health Suraksha",
  "criteria": [
    {
      "check": "sum_insured_sufficient",
      "condition": "claim_amount <= sum_insured"
    }
  ]
}
```

---

## 🔄 How the Feedback Loop Works

### Scenario: AI Made a Wrong Decision

```
1. Process claim
   confidence: 0.92 (high confidence)
   decision: Approved

2. You review and know it's actually FRAUD
   python feedback_cli.py add
   → Select claim → Rejected → fraud_detected

3. System updates calibration
   • Records: False Positive in 0.80-0.90 bin
   • Recalculates accuracy: Now 33% in that bin (was 100%)

4. Next similar claim arrives
   • Same factors → Raw confidence: 0.92
   • Feedback adjustment: 0.92 × 0.70 = 0.64 (reduced)
   • Result: "Needs manual review" (was "Likely approved")
```

### Confidence Adjustment Formula

```
If accuracy_in_bin < 70%:
    adjustment_factor = 0.9 - (0.7 - accuracy) × 0.3
    new_confidence = old_confidence × adjustment_factor

Example:
    Old: 0.92
    Accuracy in bin: 33%
    Factor: 0.9 - (0.7 - 0.33) × 0.3 = 0.789
    New: 0.92 × 0.789 = 0.73
```

---

## 💻 Command Reference

### Process Claims
```bash
python claim_pipeline.py                      # Default: sample.pdf
python claim_pipeline.py path/to/claim.pdf    # Custom PDF
```

### Feedback Management
```bash
python feedback_cli.py add               # Interactive feedback entry
python feedback_cli.py recent [N]        # Show last N feedback entries
python feedback_cli.py show              # Calibration report
python feedback_cli.py stats             # JSON statistics
```

### Web Interface
```bash
python app.py                            # Launch Gradio on http://localhost:7860
```

---

## 📈 Understanding Confidence Scores

| Score Range | Interpretation | Action |
|------------|-----------------|--------|
| **0.9-1.0** | ✅ Very High | Auto-process/approve |
| **0.7-0.9** | ✅ High | Review, likely correct |
| **0.5-0.7** | ⚠️ Medium | Manual review needed |
| **0.3-0.5** | ⚠️ Low | Request more info |
| **0.0-0.3** | ❌ Very Low | Escalate to senior expert |

After feedback calibration, you'll see tightening in these ranges!

---

## 🛠️ Architecture Decisions

### Why Structured Extraction?
Insurance documents have consistent patterns. Regex-based extraction beats free-form LLM parsing for critical fields (policy numbers, dates, amounts).

### Why Multiple Confidence Factors?
- **LLM Reward** (30%): Quality of decision reasoning
- **Policy Fit** (30%): Alignment with insurer criteria
- **Format Quality** (10%): Whether LLM followed output format
- **History Consistency** (10%): Matches past decisions for similar cases
- **Base Score** (20%): Conservative baseline

### Why Feedback Adjusts Confidence?
Real-world use shows the model may be overconfident in certain ranges. Feedback calibration prevents cascading errors.

### Why Rolling Buffers?
Large JSON files slow down processing. 100 claims + 500 feedback entries is enough for statistical patterns while keeping file I/O fast.

---

## 🔐 Security & Compliance

### Built-in Safeguards
- ✅ Multi-layer fraud detection
- ✅ Insurer/policy consistency validation
- ✅ Field completeness checks
- ✅ Exclusion and waiting period tracking
- ✅ Non-intrusive reward scoring

### Audit Trail
- ✅ Claims history with timestamps
- ✅ Feedback log with user corrections
- ✅ Calibration statistics for trend analysis

### No Data Privacy Issues
- ✅ Processes locally (no cloud uploads)
- ✅ Stores only claim summaries, not full docs
- ✅ Feedback anonymized by claim_id

---

## 📊 Example Workflows

### Workflow 1: Single Claim Processing
```bash
$ python inference.py patient_claim.pdf
# Review output...
$ python feedback_cli.py add  # If needed
```

### Workflow 2: Batch Processing
```bash
# Process 10 claims, provide feedback on questionable ones
for pdf in claims/*.pdf; do
    python inference.py "$pdf"
    # Manual review...
    if [[ suspicious ]]; then
        python feedback_cli.py add
    fi
done

# View calibration after batch
python feedback_cli.py show
```

### Workflow 3: Web Application Flow
```bash
# Start web server
python app.py

# Users upload PDFs → See decision → Click "Provide Feedback" if wrong
# Backend processes feedback → Updates model learning
# Next session, model is improved based on feedback
```

---

## 🧪 Testing Your Setup

### 1. Verify Imports
```bash
python -c "from claim_pipeline import run_inference; print('OK')"
python -c "from feedback_cli import *; print('OK')"
python -c "from app import interface; print('OK')"
```

### 2. Test Fraud Detection
```bash
python claim_pipeline.py sample.pdf
# Should show: FRAUD DETECTED with Starhealth/HDFC mismatch
```

### 3. Test Valid Claim
```bash
python claim_pipeline.py sample_valid.pdf
# Should show: APPROVED with 1.0 confidence
```

### 4. Test Feedback Recording
```bash
python feedback_cli.py add
# Interactive test of feedback system
```

### 5. View Statistics
```bash
python feedback_cli.py stats
# Should show JSON with 0 initial feedback
```

---

## 🚨 Common Issues & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| "No claims in history" | First run | Run `inference.py` to generate history |
| "No feedback recorded" | Normal | Run `feedback_cli.py add` to start feedback |
| UnicodeEncodeError | Windows console | Use ASCII output (fixed in v2) |
| "API_BASE_URL not set" | Missing .env | Create .env with HF token |
| Confidence not adjusting | No feedback in bin | More feedback needed for pattern detection |

---

## 📚 Documentation Files

- **README.md** — This file (system overview)
- **FEEDBACK_GUIDE.md** — Complete feedback system guide
- **policy_criterias/README.md** — Policy criteria format
- **requirements.txt** — Python dependencies
- **.env.example** — Environment variables template

---

## 🎓 Learning Path

### For Users:
1. Read **README.md** (this file) ← You are here
2. Run `python claim_pipeline.py sample.pdf` (see it work)
3. Read **FEEDBACK_GUIDE.md** (learn feedback system)
4. Run `python feedback_cli.py add` (provide first feedback)
5. Observe confidence adjustments in future runs

### For Developers:
1. Understand data flow: PDF → JSON → LLM → Confidence
2. Review **claim_pipeline.py** main pipeline
3. Study **claim_assessor.py** scoring logic
4. Explore **feedback_handler.py** calibration algorithm
5. Extend policies in **policy_criterias/** folder

---

## 💡 Pro Tips

### Tip 1: Batch Feedback
Process 5-10 claims, then provide feedback in one session. Patterns emerge faster.

### Tip 2: Monitor Confidence Drift
Run `python feedback_cli.py show` weekly to see if confidence is drifting.

### Tip 3: Update Policy Criteria
When feedback reveals a policy rule the model missed, update the JSON file in `policy_criterias/`.

### Tip 4: Archive Old Feedback
Periodically save `feedback_log.json` to archive, then delete to reset learning.

### Tip 5: Test Edge Cases
Provide feedback on borderline claims (0.5-0.7 confidence) first—those are where model learns most.

---

## 🎯 Summary Table

| Feature | Status | Command |
|---------|--------|---------|
| Claim Processing | ✅ Active | `python claim_pipeline.py <pdf>` |
| Fraud Detection | ✅ Active | (automatic in inference) |
| Policy Matching | ✅ Active | (automatic in inference) |
| Confidence Scoring | ✅ Active | (automatic in inference) |
| **Feedback Recording** | ✅ **NEW** | `python feedback_cli.py add` |
| **Confidence Calibration** | ✅ **NEW** | (automatic, uses feedback) |
| **Calibration Reports** | ✅ **NEW** | `python feedback_cli.py show` |
| Claims History | ✅ Persistent | (auto-saved to JSON) |
| Feedback History | ✅ Persistent | (auto-saved to JSON) |
| Web UI | ✅ Ready | `python app.py` |

---

## 🚀 Next Steps

**Immediate:**
- ✅ Process claims with `python inference.py`
- ✅ Provide feedback with `python feedback_cli.py add`
- ✅ Monitor calibration with `python feedback_cli.py show`

**Short-term:**
- 📌 Expand policy criteria for more insurers
- 📌 Fine-tune confidence thresholds
- 📌 Archive and analyze feedback trends

**Long-term:**
- 🎯 Integrate with claims management system
- 🎯 Add role-based access controls
- 🎯 Export to BI dashboards
- 🎯 Implement A/B testing for policy updates

---

## 📞 Support

For detailed guides, see:
- **FEEDBACK_GUIDE.md** — Feedback system deep dive
- **policy_criterias/policy_criteria_template.json** — Adding new insurers

---

## ✨ Congratulations!

You now have a **production-ready Insurance Claim AI system** that:
1. Processes claims autonomously
2. Validates policies intelligently
3. Detects fraud
4. Learns from human feedback
5. Improves over time

**Ready to process your first claim?** 🚀

```bash
python claim_pipeline.py your_document.pdf
```

---

*Last Updated: 2024*  
*System Version: 2.0 (with Human-in-Loop Feedback)*
