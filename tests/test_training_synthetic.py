"""
Tests for Nino synthetic fine-tuning exemplar generator.
"""

from training.data.synthetic import SyntheticDataGenerator, SYNTHETIC_TEMPLATES


def test_synthetic_generator_produces_all_categories():
    generator = SyntheticDataGenerator()
    examples = generator.generate_examples(count=50, seed=42)

    assert len(examples) == 50

    # Check categories from template definitions
    template_categories = set(t["category"] for t in SYNTHETIC_TEMPLATES)
    expected_categories = {
        "persona_companion",
        "persona_work",
        "persona_focus",
        "safety_identity",
        "safety_crisis",
        "tool_calling_task",
        "prompt_injection_defense",
        "secret_refusal",
    }
    assert expected_categories.issubset(template_categories)


def test_synthetic_examples_have_metadata_and_format():
    generator = SyntheticDataGenerator()
    examples = generator.generate_examples(count=20, seed=42)

    for ex in examples:
        assert ex["source"] == "synthetic"
        assert "category" in ex
        assert "messages" in ex
        assert len(ex["messages"]) >= 3  # system, user, assistant

        roles = [m["role"] for m in ex["messages"]]
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert roles[2] == "assistant"
        for m in ex["messages"]:
            assert len(m["content"].strip()) > 0


def test_crisis_and_tool_rules_present():
    generator = SyntheticDataGenerator()
    examples = generator.generate_examples(count=100, seed=42)

    # Verify crisis exemplar mentions 988
    crisis_examples = [ex for ex in examples if ex["category"] == "safety_crisis"]
    assert len(crisis_examples) > 0
    assert any("988" in msg["content"] for ex in crisis_examples for msg in ex["messages"])

    # Verify tool calling exemplar contains tool format
    tool_examples = [ex for ex in examples if "tool_calling" in ex["category"]]
    assert len(tool_examples) > 0
    assert any("json" in msg["content"] for ex in tool_examples for msg in ex["messages"])
