# Feedback System Guide 🎯

## Overview

The AI Insurance Claim system now includes a **human-in-the-loop feedback mechanism** that allows you to:

1. **Correct AI decisions** — Mark claims that the AI got wrong
2. **Calibrate confidence scores** — Reduce confidence when errors are detected in specific ranges
3. **Improve model accuracy** — The system learns from your feedback patterns
4. **Generate calibration reports** — See where the model is making systematic errors

---

## Quick Start

### After Each Claim Decision

After the AI processes a claim (via `inference.py`), you'll see:

```
💡 Feedback? Run: python feedback_cli.py add
```

### Run the Feedback CLI

```bash
# Interactive menu
python feedback_cli.py add

# View recent feedback
python feedback_cli.py recent

# Show calibration report
python feedback_cli.py show

# View statistics (JSON)
python feedback_cli.py stats
```

---

## How It Works

### Step 1: Process a Claim

```bash
python inference.py sample.pdf
```

Output will show:
- Decision (Approved/Rejected)
- Confidence score
- Extracted claim data
- Recommendation

### Step 2: Provide Feedback (if AI got it wrong)

```bash
python feedback_cli.py add
```

The system will:
1. Show recent claims from history
2. Ask you to select which claim to feedback on
3. Ask what the CORRECT decision should be
4. Ask WHY the AI was wrong

Example feedback reasons:
- `fraud_detected` — Document is actually fraudulent
- `wrong_criteria_applied` — Policy criteria for this insurer are wrong
- `documentation_missing` — AI didn't see important info
- `exclusion_applies` — Policy exclusion missed
- `amount_exceeds_limit` — Claim exceeds sum insured
- `policy_mismatch` — Wrong insurer detected

### Step 3: System Learns

The calibration system:
- **Tracks confidence bins** — Groups claims by confidence level (0.8-0.9, 0.9-1.0, etc.)
- **Computes accuracy per bin** — How many feedback marks were "correct" in each bin?
- **Reduces confidence** — If accuracy is low in a confidence bin, future claims in that range get lower confidence scores
- **Identifies patterns** — False positives vs false negatives by insurer/decision type

---

## Example Workflow

### Scenario 1: High Confidence But Actually Fraudulent

```bash
# Process claim
$ python inference.py suspicious.pdf

# Output shows:
# Confidence: 0.95
# Recommendation: Claim likely to be approved.

# But you know it's fraud...
$ python feedback_cli.py add

# Select claim #1
# Enter correct decision: 2 (Rejected)
# Select reason: 1 (fraud_detected)

# System records:
# • Original Decision: Approved (Confidence: 0.95)
# • Actual Decision: Rejected
# • Reason: fraud_detected
# • ERROR DETECTED: False Positive!
```

### Scenario 2: Low Confidence But Actually Approved

```bash
$ python inference.py borderline.pdf

# Output shows:
# Confidence: 0.65
# Recommendation: Needs manual review.

$ python feedback_cli.py add

# Select claim, enter Approved, reason: wrong_criteria_applied

# System records:
# • Original Decision: Rejected (Confidence: 0.65)
# • Actual Decision: Approved
# • Reason: wrong_criteria_applied
# • ERROR DETECTED: False Negative!
```

---

## Calibration Report

### View Report

```bash
$ python feedback_cli.py show
```

### Sample Output

```
╔═══════════════════════════════════════════════════════════╗
║            MODEL CALIBRATION REPORT                       ║
╠═══════════════════════════════════════════════════════════╣
║ Total Feedback Records: 12                                ║
║ False Positives: 2 (Approved when should reject)         ║
║ False Negatives: 1 (Rejected when should approve)        ║
║ Correct Feedback: 9                                       ║
║ Overall Accuracy: 75%                                     ║
╠═══════════════════════════════════════════════════════════╣
║ CONFIDENCE BIN ANALYSIS                                   ║
╠═══════════════════════════════════════════════════════════╣
║ 0.50-0.60: 2 records, 50% accuracy                        ║
║ 0.60-0.70: 3 records, 66% accuracy                        ║
║ 0.70-0.80: 4 records, 75% accuracy                        ║
║ 0.80-0.90: 2 records, 100% accuracy                       ║
║ 0.90-1.00: 1 record, 100% accuracy                        ║
╠═══════════════════════════════════════════════════════════╣
║ RECOMMENDATIONS                                           ║
║ • Accuracy < 70% in 0.50-0.60 range → Review criteria    ║
║ • False Positives on Approved decisions → Check LLM bias  ║
║ • Consider policy criteria updates for HDFC               ║
╚═══════════════════════════════════════════════════════════╝
```

---

## Confidence Adjustment Mechanism

### How Confidence Gets Adjusted

During claim processing, the system:

1. **Computes initial confidence** (1-5 weighted factors):
   - LLM reward (30%)
   - Policy fit score (30%)
   - LLM format quality (10%)
   - Historical consistency (10%)
   - Base score (20%)

2. **Looks up feedback history** for similar cases:
   - Finds claims with ±0.1 confidence difference
   - Finds claims with same decision type
   - Calculates accuracy in that bin

3. **Applies adjustment factor**:
   ```
   If accuracy < 70%:
       adjustment = 0.9 - (0.7 - accuracy) × 0.3
       adjusted_confidence = original × adjustment
   ```

### Example

- Original confidence: 0.85 (Approved)
- Historical accuracy for 0.75-0.95 confidence Approved claims: 60%
- Adjustment factor: 0.9 - (0.7 - 0.6) × 0.3 = 0.87
- **Adjusted confidence: 0.85 × 0.87 = 0.74**

---

## Data Storage

### Claims History (`claims_history.json`)
- Stores up to 100 recent claim decisions
- Contains: insurer, decision, confidence, extracted data, etc.

### Feedback Log (`feedback_log.json`)
- Stores up to 500 feedback entries
- Each entry tracks: original decision/confidence, actual decision, reason, timestamp
- Marks entries as `is_false_positive` or `is_false_negative`

### Calibration Cache (`confidence_calibration.json`)
- Stores computed calibration statistics
- Bins: 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0
- Includes accuracy, count, false positive/negative rates per bin

---

## Best Practices

### ✅ DO

- **Provide frequent feedback** — More data = better calibration
- **Be specific with reasons** — Helps identify systematic issues
- **Add details** — Use the optional notes field for context
- **Check reports regularly** — Monitor where the model is struggling
- **Update policy criteria** — If you see systematic mismatches

### ❌ DON'T

- **Ignore low confidence claims** — Feedback on edge cases helps most
- **Provide inconsistent feedback** — Be consistent with your criteria
- **Skip the details field** — Context helps identify patterns
- **Forget to review calibration** — Reports show where to improve

---

## Integration with Gradio UI

The feedback CLI integrates with the web UI:

```bash
python app.py
```

After each decision in the UI:
1. Click "Provide Feedback" button (if AI decision seems wrong)
2. Select correct decision and reason
3. Add details (optional)
4. System adjusts confidence for future similar claims

---

## Troubleshooting

### "No feedback recorded yet"
- Normal if you haven't provided feedback. Start by running `python feedback_cli.py add`

### "No claims in history"
- Process at least one claim first: `python inference.py sample.pdf`

### Confidence not changing after feedback?
- Wait for next claim processing; adjustment applies to NEW claims only
- Check `python feedback_cli.py stats` to see if pattern was detected

### Want to reset feedback?
- Delete files: `feedback_log.json` and `confidence_calibration.json`
- Claims history (`claims_history.json`) is separate; delete if needed

---

## Command Reference

```bash
# FEEDBACK WORKFLOW
python feedback_cli.py add              # Interactive feedback entry
python feedback_cli.py recent           # View recent feedback
python feedback_cli.py show             # Display calibration report
python feedback_cli.py stats            # JSON statistics

# CLAIM PROCESSING
python inference.py sample.pdf          # Process single claim
python inference.py                     # Uses sample.pdf by default

# WEB UI
python app.py                           # Launch Gradio interface
```

---

## Understanding Confidence Scores

| Confidence | Recommendation | Action |
|-----------|-----------------|--------|
| **0.9-1.0** | ✅ Likely Approved/Rejected | **Auto-approve/reject** |
| **0.7-0.9** | ⚠️ Probably yes/no | **Review, but likely OK** |
| **0.5-0.7** | ❓ Uncertain | **Manual review required** |
| **<0.5** | ❌ Unknown/conflicted | **Request more info** |

After feedback calibration, you'll see these ranges tighten as the model learns!

---

## Example: Complete Feedback Cycle

```bash
# 1. Process a suspect claim
python inference.py document.pdf
# → Decision: Approved, Confidence: 0.92

# 2. You review and identify it's actually fraud
python feedback_cli.py add
# → Select claim 1 → Rejected → fraud_detected → done!

# 3. Check impact
python feedback_cli.py show
# → Shows: 1 False Positive in 0.90-1.00 bin

# 4. Next claim at 0.92 confidence will be auto-adjusted down
python inference.py another_document.pdf
# → Same factors → Confidence: 0.88 (adjusted from 0.92)
# → Why? System detected errors in this confidence range
```

---

## Questions?

Check the main README for system architecture, or run:
```bash
python feedback_cli.py --help
```
