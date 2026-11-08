"""Interactive review CLI allowing the user to inspect, accept, reject, or edit training turns."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


class ReviewCLI:
    """Terminal review interface for human curation of training datasets."""

    def __init__(
        self,
        input_stream: Any = sys.stdin,
        output_stream: Any = sys.stdout,
        prompt_fn: Callable[[str], str] | None = None,
    ):
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.prompt_fn = prompt_fn

    def _prompt(self, msg: str) -> str:
        if self.prompt_fn is not None:
            return self.prompt_fn(msg)
        self.output_stream.write(msg)
        self.output_stream.flush()
        line = self.input_stream.readline()
        return line.strip() if line else ""

    def review_examples(
        self,
        candidate_examples: list[dict[str, Any]],
        auto_accept: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Review candidate turns with interactive human approval.

        Returns:
            (accepted_examples, rejected_examples)
        """
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        total = len(candidate_examples)
        self.output_stream.write(f"\n=== Nino Training Data Review ({total} candidate examples) ===\n")
        self.output_stream.write("Options: [A]ccept | [R]eject | [S]kip | [Q]uit\n\n")

        for idx, ex in enumerate(candidate_examples, start=1):
            if auto_accept:
                accepted.append(ex)
                continue

            messages = ex.get("messages", [])
            source = ex.get("source", "unknown")

            self.output_stream.write(f"--- Example {idx}/{total} (source: {source}) ---\n")
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                prefix = f"[{role.upper()}]: "
                self.output_stream.write(f"{prefix}{content}\n")

            while True:
                choice = self._prompt("Action ([A]ccept, [R]eject, [S]kip, [Q]uit): ").lower()
                if choice in ("a", "accept", ""):
                    accepted.append(ex)
                    self.output_stream.write("  -> Accepted.\n\n")
                    break
                elif choice in ("r", "reject"):
                    rejected.append(ex)
                    self.output_stream.write("  -> Rejected.\n\n")
                    break
                elif choice in ("s", "skip"):
                    self.output_stream.write("  -> Skipped.\n\n")
                    break
                elif choice in ("q", "quit"):
                    self.output_stream.write("  -> Quitting review session early.\n\n")
                    return accepted, rejected
                else:
                    self.output_stream.write("  Invalid choice. Enter 'a', 'r', 's', or 'q'.\n")

        self.output_stream.write(
            f"\nReview complete: {len(accepted)} accepted, {len(rejected)} rejected out of {total}.\n"
        )
        return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description="Review and curate training data for Nino fine-tuning.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to unreviewed JSONL file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to write accepted JSONL file")
    parser.add_argument("--auto-accept", action="store_true", help="Non-interactive: accept all clean candidates")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input file {in_path} does not exist.")
        sys.exit(1)

    candidates: list[dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    reviewer = ReviewCLI()
    accepted, _ = reviewer.review_examples(candidates, auto_accept=args.auto_accept)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in accepted:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(accepted)} approved examples to {out_path}.")


if __name__ == "__main__":
    main()
