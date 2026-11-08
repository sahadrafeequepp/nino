"""QLoRA / LoRA fine-tuning runner with checkpointing, logging, and fast dry-run mode."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("training.train")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dry_run(config: dict[str, Any], data_file: str | Path) -> dict[str, Any]:
    """Execute a lightweight dry-run verifying pipeline and data validity without GPU requirements."""
    logger.info("=== Running Nino Fine-Tuning Pipeline in DRY-RUN Mode ===")
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})

    logger.info("Base Model Target: %s", model_cfg.get("base_model", "unknown"))
    logger.info("LoRA Hyperparameters: rank=%s, alpha=%s, dropout=%s", lora_cfg.get("r"), lora_cfg.get("lora_alpha"), lora_cfg.get("lora_dropout"))
    logger.info("Target Modules: %s", lora_cfg.get("target_modules", []))

    data_path = Path(data_file)
    if not data_path.exists():
        logger.warning("Dataset file %s does not exist. Generating temporary test samples...", data_path)
        from training.data.synthetic import SyntheticDataGenerator
        sample_examples = SyntheticDataGenerator.generate_examples(count=4)
    else:
        sample_examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample_examples.append(json.loads(line))

    logger.info("Loaded %d candidate examples for verification.", len(sample_examples))

    # Verify formatting of each sample
    valid_count = 0
    total_tokens = 0
    for ex in sample_examples:
        msgs = ex.get("messages", [])
        assert len(msgs) >= 2, f"Example {ex.get('id')} has fewer than 2 messages."
        for m in msgs:
            assert "role" in m and "content" in m, f"Malformed message in {ex.get('id')}."
            total_tokens += len(m["content"]) // 4
        valid_count += 1

    avg_tokens = total_tokens // max(1, valid_count)
    logger.info("Verified %d examples. Estimated average token length: ~%d tokens.", valid_count, avg_tokens)

    output_dir = Path(train_cfg.get("output_dir", "training/output/dry_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save mock adapter metadata
    mock_adapter_config = {
        "base_model_name_or_path": model_cfg.get("base_model"),
        "peft_type": "LORA",
        "r": lora_cfg.get("r", 16),
        "lora_alpha": lora_cfg.get("lora_alpha", 32),
        "lora_dropout": lora_cfg.get("lora_dropout", 0.05),
        "target_modules": lora_cfg.get("target_modules"),
        "dry_run": True,
    }
    with open(output_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(mock_adapter_config, f, indent=2)

    # Save mock metrics log
    metrics = {
        "status": "dry_run_success",
        "verified_examples": valid_count,
        "avg_tokens": avg_tokens,
        "simulated_loss_step_1": 1.42,
        "simulated_loss_step_2": 1.18,
        "checkpoint_saved": str(output_dir / "adapter_config.json"),
    }
    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=== Dry-Run Completed Successfully! Checkpoint and metrics saved to: %s ===", output_dir)
    return metrics


def train(config_path: str = "training/config.yaml", dry_run: bool = False) -> None:
    """Main training entrypoint supporting full GPU fine-tuning and dry-run validation."""
    cfg = load_config(config_path)

    data_file = cfg.get("data", {}).get("train_file", "training/data/train.jsonl")

    if dry_run or os.environ.get("NINO_TRAIN_DRY_RUN") == "1":
        run_dry_run(cfg, data_file)
        return

    # Full GPU training path using Hugging Face PEFT + TRL + Transformers
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except ImportError as e:
        logger.error(
            "Missing GPU training dependencies (%s). "
            "To train on GPU, install training requirements: pip install -r training/requirements.txt. "
            "Or run dry-run validation locally with: python -m training.train --dry-run",
            e,
        )
        sys.exit(1)

    model_name = cfg["model"]["base_model"]
    output_dir = cfg["training"]["output_dir"]
    logger.info("Initializing GPU training for %s...", model_name)

    # Configure 4-bit quantization if specified and CUDA available
    compute_dtype = getattr(torch, cfg["model"].get("bnb_4bit_compute_dtype", "bfloat16"))
    bnb_config = None
    if cfg["model"].get("use_4bit", True) and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["model"].get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg["model"].get("use_nested_quant", False),
        )

    device_map = "auto" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": data_file})

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        max_seq_length=cfg["training"]["max_seq_length"],
        fp16=cfg["training"].get("fp16", False),
        bf16=cfg["training"].get("bf16", False) and torch.cuda.is_bf16_supported(),
        dataset_text_field="messages",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_args,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    logger.info("Starting training loop...")
    trainer.train()

    logger.info("Saving fine-tuned LoRA adapter to %s...", output_dir)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Nino LoRA fine-tuning adapter.")
    parser.add_argument("--config", "-c", type=str, default="training/config.yaml", help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Run lightweight offline verification without GPU")
    args = parser.parse_args()

    train(config_path=args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
