"""
Tests for Nino training pipeline dry-run, Modelfile packaging, and rollback.
"""

import json
from pathlib import Path
from training.package import generate_modelfile, switch_active_model, rollback_to_base
from training.train import run_dry_run


def test_dry_run_pipeline(tmp_path):
    # Prepare dummy train jsonl file
    train_file = tmp_path / "train.jsonl"
    output_dir = tmp_path / "outputs"

    dummy_data = [
        {
            "id": "dry_001",
            "source": "synthetic",
            "messages": [
                {"role": "user", "content": "What is the time?"},
                {"role": "assistant", "content": "It's 3:00 PM."},
            ],
        }
    ]

    with open(train_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(dummy_data[0]) + "\n")

    config = {
        "model": {
            "base_model": "google/gemma-2-2b-it",
            "use_4bit": True,
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "training": {
            "output_dir": str(output_dir),
        },
    }

    # Run dry-run execution
    metrics = run_dry_run(
        config=config,
        data_file=train_file,
    )
    assert metrics["status"] == "dry_run_success"
    assert metrics["verified_examples"] == 1

    # Check mock adapter checkpoint was written
    assert output_dir.exists()
    assert (output_dir / "adapter_config.json").exists()
    assert (output_dir / "training_metrics.json").exists()


def test_modelfile_generation(tmp_path):
    modelfile_path = tmp_path / "Modelfile"
    content = generate_modelfile(
        base_model="gemma2:2b",
        adapter_path="./outputs/final_adapter",
        system_prompt="Test persona prompt",
        temperature=0.6,
        top_p=0.85,
        output_path=str(modelfile_path),
    )

    assert modelfile_path.exists()
    assert "FROM gemma2:2b" in content
    assert "ADAPTER ./outputs/final_adapter" in content
    assert "PARAMETER temperature 0.6" in content
    assert "PARAMETER top_p 0.85" in content
    assert "Test persona prompt" in content


def test_switch_model_and_rollback(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("API_PORT=8000\nLLM_MODEL=gemma2:2b\n", encoding="utf-8")

    # Switch to fine-tuned model
    switch_active_model("nino:finetuned", env_file_path=str(env_file))
    content = env_file.read_text(encoding="utf-8")
    assert "LLM_MODEL=nino:finetuned" in content

    # Roll back to base model
    rollback_to_base(base_model="gemma2:2b", env_file_path=str(env_file))
    content_after = env_file.read_text(encoding="utf-8")
    assert "LLM_MODEL=gemma2:2b" in content_after
