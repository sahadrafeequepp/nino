"""
Package and deploy fine-tuned LoRA adapters to Ollama with instant rollback.

This utility takes a fine-tuned LoRA adapter directory, verifies its artifacts,
generates a custom Ollama Modelfile, and provides commands/helpers to build
and activate the custom model in Ollama, or instantly rollback to the base model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def generate_modelfile(
    base_model: str = "gemma2:2b",
    adapter_path: str = "./outputs/final_adapter",
    system_prompt: str = "You are Nino, a warm, observant, and thoughtful personal companion assistant.",
    temperature: float = 0.7,
    top_p: float = 0.9,
    output_path: str = "Modelfile",
) -> str:
    """Generate an Ollama Modelfile pointing to the base model and adapter.

    Args:
        base_model: Ollama base model tag (e.g. gemma2:2b).
        adapter_path: Absolute or relative path to LoRA adapter weights (e.g. ./final_adapter).
        system_prompt: Default system prompt for Nino.
        temperature: Sampling temperature.
        top_p: Top-p nucleus sampling parameter.
        output_path: Destination path for the generated Modelfile.

    Returns:
        The content of the generated Modelfile.
    """
    # Normalize adapter path to forward slashes for cross-platform Ollama compatibility
    clean_adapter_path = adapter_path.replace("\\", "/")

    modelfile_content = f"""# Nino Fine-Tuned Persona & Tool-Calling Model
# Generated automatically by training/package.py

FROM {base_model}

# Attach fine-tuned LoRA adapter weights
ADAPTER {clean_adapter_path}

# Model parameters tuned for conversation consistency
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"

# Nino Core Persona System Prompt
SYSTEM \"\"\"
{system_prompt.strip()}
\"\"\"
"""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    return modelfile_content


def verify_adapter(adapter_dir: Path) -> bool:
    """Check if the adapter directory contains necessary PEFT artifacts."""
    if not adapter_dir.exists():
        return False
    required_files = ["adapter_config.json"]
    has_weights = (
        (adapter_dir / "adapter_model.safetensors").exists()
        or (adapter_dir / "adapter_model.bin").exists()
    )
    has_config = (adapter_dir / "adapter_config.json").exists()
    return has_config and has_weights


def build_ollama_model(
    model_name: str = "nino:finetuned",
    modelfile_path: str = "Modelfile",
) -> bool:
    """Invoke ollama create <model_name> -f <modelfile_path>."""
    cmd = ["ollama", "create", model_name, "-f", modelfile_path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[+] Successfully built Ollama model '{model_name}':\n{res.stdout}")
        return True
    except FileNotFoundError:
        print("[-] 'ollama' executable not found in PATH. Install Ollama or run manually.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[-] Failed to build Ollama model '{model_name}':\n{e.stderr}")
        return False


def switch_active_model(
    target_model: str,
    env_file_path: str = ".env",
) -> bool:
    """Switch LLM_MODEL in .env file or report configuration update.

    Args:
        target_model: Model name to set (e.g. 'nino:finetuned' or 'gemma2:2b' for rollback).
        env_file_path: Path to the environment configuration file.

    Returns:
        True if updated or created, False otherwise.
    """
    env_path = Path(env_file_path)
    lines = []
    found = False

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("LLM_MODEL="):
                    lines.append(f"LLM_MODEL={target_model}\n")
                    found = True
                else:
                    lines.append(line)

    if not found:
        lines.append(f"LLM_MODEL={target_model}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[+] Switched active LLM_MODEL to '{target_model}' in {env_file_path}")
    return True


def rollback_to_base(base_model: str = "gemma2:2b", env_file_path: str = ".env") -> bool:
    """Instantly roll back the assistant to the base model."""
    print(f"[*] Initiating instant rollback to base model '{base_model}'...")
    return switch_active_model(base_model, env_file_path=env_file_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Package and deploy fine-tuned Nino model to Ollama")
    parser.add_argument(
        "--action",
        choices=["modelfile", "build", "activate", "rollback"],
        default="modelfile",
        help="Action to perform: generate Modelfile, build Ollama model, activate in .env, or rollback",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="./training/outputs/final_adapter",
        help="Path to fine-tuned LoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="gemma2:2b",
        help="Base model tag in Ollama",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nino:finetuned",
        help="Target Ollama model name",
    )
    parser.add_argument(
        "--modelfile-path",
        type=str,
        default="training/Modelfile",
        help="Path to write the Modelfile",
    )
    parser.add_argument(
        "--env-path",
        type=str,
        default=".env",
        help="Path to .env file for active model switching",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.action == "modelfile":
        content = generate_modelfile(
            base_model=args.base_model,
            adapter_path=args.adapter_dir,
            output_path=args.modelfile_path,
        )
        print(f"[+] Modelfile generated at {args.modelfile_path}:\n")
        print(content)

    elif args.action == "build":
        # First ensure Modelfile exists
        generate_modelfile(
            base_model=args.base_model,
            adapter_path=args.adapter_dir,
            output_path=args.modelfile_path,
        )
        build_ollama_model(model_name=args.model_name, modelfile_path=args.modelfile_path)

    elif args.action == "activate":
        switch_active_model(args.model_name, env_file_path=args.env_path)
        print(f"[+] Nino will now use '{args.model_name}'. Restart the assistant to apply.")

    elif args.action == "rollback":
        rollback_to_base(base_model=args.base_model, env_file_path=args.env_path)
        print(f"[+] Nino is rolled back to '{args.base_model}'. Restart the assistant to apply.")


if __name__ == "__main__":
    main()
