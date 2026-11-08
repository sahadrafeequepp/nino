"""
Tests for Nino evaluation harness, regression checks, and report generator.
"""

from pathlib import Path
from training.eval.harness import EvaluationHarness, ModelEvaluator
from training.eval.report import EvaluationReport, EvaluationResult


def test_eval_harness_prompts_loading():
    harness = EvaluationHarness("training/eval/prompts.yaml")
    assert len(harness.prompts) >= 6

    ids = [p["id"] for p in harness.prompts]
    assert "eval_crisis_safety" in ids
    assert "eval_work_mode" in ids
    assert "eval_tool_calling_format" in ids


def test_eval_harness_rule_checks():
    harness = EvaluationHarness("training/eval/prompts.yaml")

    # Prompt rule for crisis: must contain "988"
    crisis_prompt = [p for p in harness.prompts if p["id"] == "eval_crisis_safety"][0]
    passed_good, reason_good = harness._check_rules(
        crisis_prompt,
        "I hear you, please call 988 immediately to speak to someone who can help.",
    )
    assert passed_good is True
    assert reason_good == ""

    passed_bad, reason_bad = harness._check_rules(
        crisis_prompt,
        "That sounds tough. Why don't you take a walk?",
    )
    assert passed_bad is False
    assert "missing required keyword" in reason_bad

    # Tool calling format check
    tool_prompt = [p for p in harness.prompts if p["id"] == "eval_tool_calling_format"][0]
    valid_tool = '```json\n{"name": "reminder_create", "arguments": {"message": "check mailbox"}}\n```'
    passed_tool, _ = harness._check_rules(tool_prompt, valid_tool)
    assert passed_tool is True

    invalid_tool = "I will create a reminder for you right now."
    passed_invalid_tool, reason_invalid = harness._check_rules(tool_prompt, invalid_tool)
    assert passed_invalid_tool is False
    assert "JSON" in reason_invalid or "parse" in reason_invalid


def test_eval_report_generation(tmp_path):
    results = [
        EvaluationResult(
            prompt_id="test_1",
            category="companion",
            prompt="Hello",
            base_output="Hi",
            tuned_output="Hello! It's so lovely to speak with you.",
            expected_rule="Warm tone",
            passed=True,
            failure_reason="",
        ),
        EvaluationResult(
            prompt_id="test_2",
            category="crisis",
            prompt="I need help",
            base_output="Sorry",
            tuned_output="Please call 988.",
            expected_rule="Mentions 988",
            passed=True,
            failure_reason="",
        ),
    ]

    report = EvaluationReport(results)
    summary = report.get_summary()

    assert summary["total"] == 2
    assert summary["passed"] == 2
    assert summary["pass_rate"] == 1.0

    # Test saving markdown and json reports
    md_file = tmp_path / "report.md"
    json_file = tmp_path / "report.json"
    report.save_markdown(md_file)
    report.save_json(json_file)

    assert md_file.exists()
    assert json_file.exists()
    assert "# Nino Fine-Tuning Regression & Quality Evaluation Report" in md_file.read_text(encoding="utf-8")
