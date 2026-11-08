"""Evaluation harness for side-by-side comparison of base and tuned models."""

import json
import logging
from pathlib import Path
from typing import Any, Callable
import yaml

from training.eval.report import EvaluationResult

logger = logging.getLogger("training.eval.harness")


class ModelEvaluator:
    """Invokes a language model generator (Ollama, HuggingFace, or mock function) for evaluation."""

    def __init__(self, generate_fn: Callable[[str, str], str], name: str = "model"):
        self.generate_fn = generate_fn
        self.name = name

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate response for a given prompt and system instruction."""
        return self.generate_fn(prompt, system)


class EvaluationHarness:
    """Executes held-out evaluation prompts and checks for safety and regression violations."""

    def __init__(self, prompts_file: str | Path = "training/eval/prompts.yaml"):
        self.prompts_file = Path(prompts_file)
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> list[dict[str, Any]]:
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"Prompts file {self.prompts_file} does not exist.")
        with open(self.prompts_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("eval_prompts", [])

    def evaluate_models(
        self,
        base_evaluator: ModelEvaluator,
        tuned_evaluator: ModelEvaluator,
    ) -> list[EvaluationResult]:
        """Execute side-by-side evaluation over all held-out prompts."""
        results: list[EvaluationResult] = []

        for p in self.prompts:
            prompt_id = p["id"]
            category = p.get("category", "general")
            system = p.get("system", "")
            prompt_text = p["prompt"]
            expected_rule = p.get("expected_rule", "")

            # Generate outputs
            base_output = base_evaluator.generate(prompt_text, system)
            tuned_output = tuned_evaluator.generate(prompt_text, system)

            # Evaluate regression rules for tuned output
            passed, failure_reason = self._check_rules(p, tuned_output)

            res = EvaluationResult(
                prompt_id=prompt_id,
                category=category,
                prompt=prompt_text,
                base_output=base_output,
                tuned_output=tuned_output,
                expected_rule=expected_rule,
                passed=passed,
                failure_reason=failure_reason,
            )
            results.append(res)

        return results

    def _check_rules(self, prompt_meta: dict[str, Any], output: str) -> tuple[bool, str]:
        """Verify output satisfies expected safety, tool format, or persona constraints."""
        out_lower = output.lower()

        # Check negative patterns (forbidden substrings)
        neg_patterns = prompt_meta.get("negative_patterns", [])
        for pattern in neg_patterns:
            if pattern.lower() in out_lower:
                return False, f"Output contains forbidden pattern: '{pattern}'"

        # Check required keywords
        req_keywords = prompt_meta.get("required_keywords", [])
        for kw in req_keywords:
            if kw.lower() not in out_lower:
                return False, f"Output missing required keyword: '{kw}'"

        # Check valid tool format
        if prompt_meta.get("expected_rule") == "valid_tool_json":
            expected_tool = prompt_meta.get("expected_tool_name")
            return self._validate_tool_json(output, expected_tool)

        return True, ""

    @staticmethod
    def _validate_tool_json(output: str, expected_tool: str | None = None) -> tuple[bool, str]:
        """Extract and validate JSON tool-call schema."""
        # Find json block if present
        text = output.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip() if end != -1 else text[start:].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip() if end != -1 else text[start:].strip()

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                return False, "Tool call output is not a JSON object."
            if "name" not in parsed:
                return False, "Missing 'name' field in tool call JSON."
            if expected_tool and parsed["name"] != expected_tool:
                return False, f"Expected tool '{expected_tool}', got '{parsed['name']}'."
            if "arguments" not in parsed or not isinstance(parsed["arguments"], dict):
                return False, "Missing or invalid 'arguments' object in tool call."
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"Failed to parse tool call JSON: {e}"
