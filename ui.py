"""Gradio UI layout for the Insurance Claim AI system.

All handler functions are injected via create_ui() - no logic lives here.
"""

import gradio as gr
from pathlib import Path

_EMPTY = ""  # sentinel for blank Markdown initial values
_ICONS = Path(__file__).parent / "icons"


def create_ui(
    process_claim,
    clear_outputs,
    reset_system,
    get_state_snapshot,
    submit_feedback,
    clear_feedback,
    refresh_reports,
    theme,
):
    """Build and return the Gradio Blocks app.

    Parameters
    ----------
    process_claim, clear_outputs, reset_system, get_state_snapshot, submit_feedback, clear_feedback, refresh_reports :
        Handler functions injected from app.py.
    theme : gr.Theme
        Gradio theme - pass to demo.launch(theme=...) in the caller.
    """
    custom_css = """
.clear-btn-style {
    background-color: red !important;
    color: #2563eb !important;
    border-color: #dbeafe !important;
    padding: 4px 12px !important;
}

.footer-text {
    text-align: left !important;
    font-size: 0.9em;
    color: gray;
    margin-top: 60px;
}

/* ✅ RESET BUTTON FULL RED */
#custom-red-btn-1 button {
    background-color: #dc2626 !important;
    color: #ffffff !important;
    border: none !important;
}

/* hover */
#custom-red-btn-1 button:hover {
    background-color: #b91c1c !important;
}

/* force icon/text white */
#custom-red-btn-1 button * {
    color: #ffffff !important;
}

#big-bold-accordion,
#big-bold-accordion button,
#big-bold-accordion span,
#big-bold-accordion .label-wrap {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}

"""
    with gr.Blocks(title="Insurance Claim AI") as demo:

        gr.Markdown(
            """
            # Insurance Claim AI Decision System
            Upload insurance claim PDFs, get AI-powered decisions with confidence scoring,
            fraud detection, and policy matching - then **train the model** with your feedback.
            """,
        )
        gr.Markdown("---")

        with gr.Tabs():
            with gr.Tab("Process Claim", id="tab_process"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        pdf_input = gr.File(
                            label="Claim PDF",
                            file_types=[".pdf"],
                            type="filepath",
                        )
                        with gr.Row(equal_height=True):
                            process_btn = gr.Button(
                                "Process Claim",
                                variant="primary",
                                size="lg",
                                scale=4
                            )
                            clear_btn = gr.Button(
                                "Clear",
                                icon=str(_ICONS / "trash.svg"),
                                variant="secondary",
                                size="lg",
                                scale=1
                            )

                    with gr.Column(scale=1):
                        quick_info_bar = gr.Textbox(
                            label="Quick Summary",
                            interactive=False,
                            lines=1,
                            placeholder="Process a claim to see the summary here...",
                        )

                with gr.Column(visible=False) as results_col:
                    gr.Markdown("### Results")

                    with gr.Tab("Details"):
                        decision_hero_md = gr.HTML(value=_EMPTY)
                        quick_insights_md = gr.HTML(value=_EMPTY)
                        fraud_md = gr.HTML(value=_EMPTY)
                        recommendation_md = gr.HTML(value=_EMPTY)
                        gr.Markdown("#### Key Metrics")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, min_width=160):
                                metrics_col1_md = gr.HTML(value=_EMPTY)
                            with gr.Column(scale=1, min_width=160):
                                metrics_col2_md = gr.HTML(value=_EMPTY)
                            with gr.Column(scale=1, min_width=160):
                                metrics_col3_md = gr.HTML(value=_EMPTY)

                        checks_summary_md = gr.Markdown(value=_EMPTY)
                        with gr.Accordion("View Detailed Policy Checks", open=False):
                            checks_md = gr.Markdown(value=_EMPTY)
                        feedback_override_md = gr.Markdown(value=_EMPTY)

                gr.Markdown("---")
                
                with gr.Accordion("System Controls & State", open=False, elem_id="big-bold-accordion"):
                    with gr.Row(equal_height=True):
                        reset_btn = gr.Button("Reset System", icon=str(_ICONS / "rotate.svg"), variant="primary", size="lg", scale=1, elem_id="custom-red-btn-1")
                        state_btn = gr.Button("Show State", icon=str(_ICONS / "eye.svg"), variant="secondary", size="lg", scale=1)
                    state_md = gr.Markdown(value="", elem_classes=["subtitle-text"])
                    state_json = gr.JSON(label="State Snapshot", visible=False)

                all_outputs = [
                    decision_hero_md,
                    quick_insights_md,
                    metrics_col1_md,
                    metrics_col2_md,
                    metrics_col3_md,
                    fraud_md,
                    recommendation_md,

                    checks_summary_md,
                    checks_md,
                    feedback_override_md,
                    quick_info_bar,
                    results_col,
                ]

                process_btn.click(
                    fn=process_claim,
                    inputs=[pdf_input],
                    outputs=all_outputs,
                    show_progress="hidden",
                )
                clear_btn.click(
                    fn=clear_outputs,
                    outputs=[pdf_input] + all_outputs,
                    show_progress="hidden",
                )
                reset_btn.click(
                    fn=reset_system,
                    outputs=all_outputs,
                    show_progress="hidden",
                )
                state_btn.click(
                    fn=get_state_snapshot,
                    outputs=[state_md, state_json],
                )

            with gr.Tab("Provide Feedback", id="tab_feedback"):
                gr.Markdown(
                    "### Correct the AI Decision\n"
                    "If the AI made a mistake on the last processed claim, "
                    "use this form to record the correction.",
                )
                gr.Markdown("---")

                gr.Markdown("### Claim Feedback Submission")
                with gr.Group():
                    correct_decision = gr.Radio(
                        choices=["Approved", "Rejected"],
                        label="What should the correct decision be?",
                        info="Select the decision that should have been made.",
                    )

                    feedback_reason = gr.Dropdown(
                        choices=[
                            "Policy mismatch",
                            "Fraud misclassification",
                            "Missing data",
                            "Incorrect validation",
                            "Other",
                        ],
                        label="Why was the AI wrong?",
                        value="Policy mismatch",
                        info="Select the primary reason for the correction.",
                    )

                    feedback_details = gr.Textbox(
                        label="Additional Details (optional)",
                        placeholder="Provide any extra context about why this decision is wrong...",
                        lines=4,
                    )

                with gr.Row(equal_height=True):
                    submit_feedback_btn = gr.Button(
                        "Submit Feedback",
                        icon=str(_ICONS / "check.svg"),
                        variant="primary",
                        size="lg",
                        scale=4
                    )
                    clear_feedback_btn = gr.Button(
                        "Reset Form",
                        icon=str(_ICONS / "rotate.svg"),
                        variant="secondary",
                        size="lg",
                        scale=1
                    )

                feedback_output = gr.Markdown(
                    value="",
                    label="Feedback Status",
                )

                submit_feedback_btn.click(
                    fn=submit_feedback,
                    inputs=[correct_decision, feedback_reason, feedback_details],
                    outputs=[feedback_output],
                )
                clear_feedback_btn.click(
                    fn=clear_feedback,
                    outputs=[correct_decision, feedback_reason, feedback_details, feedback_output],
                )

            with gr.Tab("Calibration", id="tab_calibration"):
                gr.Markdown(
                    "### Model Performance & Calibration\n"
                    "Click **Refresh** to see how the model is learning from corrections.",
                )

                refresh_btn = gr.Button(
                    "Refresh Report",
                    icon=str(_ICONS / "refresh.svg"),
                    variant="primary",
                    size="lg",
                )

                report_output = gr.Markdown(
                    value="*Click **Refresh Report** to load calibration data.*",
                    label="Calibration Report",
                )
                with gr.Accordion("View Raw Data", open=False):
                    stats_output = gr.JSON(
                        label="Raw Statistics",
                    )

                refresh_btn.click(
                    fn=refresh_reports,
                    outputs=[report_output, stats_output],
                )

            with gr.Tab("Help & Guide", id="tab_help"):
                gr.Markdown(
                    """
## How to Use This System

### 1. Process Claims
1. Go to the **Process Claim** tab
2. Upload an insurance claim PDF
3. Click **Process Claim**
4. Review the structured decision, fraud check, and extracted data

### 2. Provide Feedback
If the AI got it wrong:
1. Go to the **Provide Feedback** tab
2. Select the **correct** decision (Approved / Rejected)
3. Choose the reason why the AI was wrong
4. Optionally add details
5. Click **Submit Feedback**

### 3. Monitor Calibration
1. Go to the **Calibration Report** tab
2. Click **Refresh Report**
3. Review accuracy, false-positive/negative rates, and confidence-bin calibration
                    """
                )

                with gr.Accordion("Understanding Confidence Scores", open=False):
                    gr.Markdown(
                        """
| Score | Meaning | Recommended Action |
|-------|---------|--------------------|
| **0.9 - 1.0** | Very high confidence | Auto-approve / reject |
| **0.7 - 0.9** | High confidence | Review likely OK |
| **0.5 - 0.7** | Medium confidence | Manual review needed |
| **< 0.5** | Low confidence | Request more info |
                        """
                    )

                with gr.Accordion("How the Model Learns", open=False):
                    gr.Markdown(
                        """
1. You provide feedback on a claim decision.
2. The system records: *"Original Approved (0.92) but actually Rejected"*.
3. It detects that in the 0.90-1.00 confidence range, accuracy is low.
4. On the next similar claim, confidence is **auto-adjusted lower** = more caution.
                        """
                    )

                with gr.Accordion("CLI Alternative", open=False):
                    gr.Markdown(
                        """
```bash
python feedback_cli.py add     # Interactive feedback
python feedback_cli.py show    # View calibration report
python feedback_cli.py stats   # Raw JSON statistics
```
                        """
                    )




    return demo, custom_css
