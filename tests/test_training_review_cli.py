"""
Tests for Nino interactive training review CLI.
"""

import io
from training.data.review_cli import ReviewCLI


def test_review_cli_accept_and_reject():
    candidates = [
        {
            "source": "synthetic",
            "messages": [
                {"role": "user", "content": "Hello Nino"},
                {"role": "assistant", "content": "Hello! How can I help today?"},
            ],
        },
        {
            "source": "synthetic",
            "messages": [
                {"role": "user", "content": "Give me a recipe"},
                {"role": "assistant", "content": "Here is a recipe..."},
            ],
        },
    ]

    # Test prompt responses: accept first, reject second
    responses = iter(["a", "r"])
    reviewer = ReviewCLI(prompt_fn=lambda prompt: next(responses))

    accepted, rejected = reviewer.review_examples(candidates)

    assert len(accepted) == 1
    assert len(rejected) == 1
    assert accepted[0]["messages"][0]["content"] == "Hello Nino"
    assert rejected[0]["messages"][0]["content"] == "Give me a recipe"


def test_review_cli_skip_and_quit():
    candidates = [
        {"source": "test", "messages": [{"role": "user", "content": "Skip me"}]},
        {"source": "test", "messages": [{"role": "user", "content": "Quit on me"}]},
        {"source": "test", "messages": [{"role": "user", "content": "Never reached"}]},
    ]

    responses = iter(["s", "q"])
    reviewer = ReviewCLI(prompt_fn=lambda prompt: next(responses))

    accepted, rejected = reviewer.review_examples(candidates)

    assert len(accepted) == 0
    assert len(rejected) == 0


def test_review_cli_auto_accept():
    candidates = [
        {"source": "test", "messages": [{"role": "user", "content": f"msg {i}"}]}
        for i in range(5)
    ]
    reviewer = ReviewCLI()
    accepted, rejected = reviewer.review_examples(candidates, auto_accept=True)

    assert len(accepted) == 5
    assert len(rejected) == 0
