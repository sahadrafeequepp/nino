# Nino LoRA Fine-Tuning Pipeline (Phase 6B)

This guide documents the fine-tuning pipeline for **Nino**, designed to teach a small open-weight language model Nino's conversational style, persona consistency, ethical boundaries, and tool-calling format.

> [!IMPORTANT]
> **Privacy Invariant**: Personal facts about the user stay strictly in the SQLite memory system (`data/nino.db`), **never** in model weights. Fine-tuning adjusts tone, persona modes, and tool-calling syntax only.

---

## 1. Hardware & Compute Requirements

### Local Machine vs. Cloud GPU
- **Local Machine**: Intel Core i3-1305U CPU with integrated Intel UHD Graphics and 8GB RAM.
  - Fine-tuning a 2B parameter model locally on CPU is memory-prohibitive and impractically slow.
  - **Local Role**: Dataset building, PII scrubbing, interactive human review, test verification, and lightweight pipeline dry-runs (`--dry-run`).
- **Target Training Hardware**:
  - **Minimum**: NVIDIA GPU with >= 8GB VRAM (NVIDIA RTX 3060/4060, T4 on Google Colab).
  - **Recommended**: NVIDIA GPU with >= 16GB VRAM (RTX 4090, A10G, A100 on RunPod or Vast.ai).
  - **Training Duration**: ~15–30 minutes for 500–1000 examples with 4-bit QLoRA.

### Recommended Base Model & Stack
- **Base Model**: `google/gemma-2-2b-it` (matches Nino's default Ollama model `gemma2:2b`).
- **Framework**: Hugging Face PEFT + TRL (`SFTTrainer`) + `bitsandbytes` (4-bit NF4 quantization) or Unsloth for ultra-fast training.

---

## 2. Directory Structure (`training/`)

All fine-tuning code is isolated in the top-level `training/` folder to prevent affecting the running assistant:

```
training/
├── __init__.py
├── requirements.txt         # GPU training dependencies (PyTorch, PEFT, TRL, bitsandbytes)
├── config.yaml              # Hyperparameters (rank, alpha, learning rate, batch size)
├── package.py               # Ollama Modelfile generator & instant rollback switcher
├── train.py                 # Training runner (supports full GPU training and local --dry-run)
├── data/
│   ├── __init__.py
│   ├── filters.py           # Secret detection, PII scrubbing, sensitive memory rejection
│   ├── exporter.py          # Opt-in session exporter and train/validation splitter
│   ├── synthetic.py         # Balanced synthetic exemplar generator
│   └── review_cli.py        # Terminal review interface for human curation
└── eval/
    ├── __init__.py
    ├── prompts.yaml         # Held-out evaluation prompts and regression rules
    ├── harness.py           # Side-by-side evaluation harness
    └── report.py            # Markdown and JSON evaluation report generator
```

---

## 3. Step-by-Step Training Guide

### Step 1: Prepare the Dataset

#### A. Export Opted-in Sessions
Nino **never** exports conversations by default. Only sessions explicitly opted in by the user are candidates:

```bash
# Export approved sessions (filtered through SecretFilter, PIIScrubber, SensitiveMemoryFilter)
python -m training.data.exporter --sessions sess_1234,sess_5678 --output training/data/candidates.jsonl
```

#### B. Generate Synthetic Exemplars
Generate high-quality exemplars for persona modes (companion, work, focus), ethical boundaries, crisis safety (988), and tool-calling JSON schemas:

```bash
python -m training.data.synthetic --count 100 --output training/data/synthetic.jsonl
```

#### C. Interactive Human Review
Inspect every candidate turn before it enters the training set:

```bash
python -m training.data.review_cli -i training/data/candidates.jsonl -o training/data/reviewed.jsonl
```
- Press `[A]` to accept.
- Press `[R]` to reject.
- Press `[S]` to skip.
- Press `[Q]` to quit review session.

#### D. Split Train & Validation Sets
Split reviewed data into 90% train and 10% validation sets:
```bash
python -m training.data.exporter --split --input training/data/reviewed.jsonl --train training/data/train.jsonl --val training/data/val.jsonl
```

---

### Step 2: Validate Locally with Dry-Run Mode

Before renting cloud compute or copying files, verify dataset schema, token counts, and training config on your local machine with zero GPU dependencies:

```bash
.venv\Scripts\python -m training.train --dry-run
```
Output:
```
[INFO] === Running Nino Fine-Tuning Pipeline in DRY-RUN Mode ===
[INFO] Base Model Target: google/gemma-2-2b-it
[INFO] LoRA Hyperparameters: rank=16, alpha=32, dropout=0.05
[INFO] Target Modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
[INFO] Verified examples. Estimated average token length: ~140 tokens.
[INFO] === Dry-Run Completed Successfully! Checkpoint and metrics saved. ===
```

---

### Step 3: Train on GPU (RunPod / Colab / Workstation)

Install GPU dependencies in your GPU environment:
```bash
pip install -r training/requirements.txt
```

Launch QLoRA fine-tuning:
```bash
python -m training.train --config training/config.yaml
```

The script will save the final adapter to `training/outputs/final_adapter/` containing `adapter_config.json` and `adapter_model.safetensors`.

---

### Step 4: Run Side-by-Side Evaluation & Regression Testing

Test the tuned model against the base model using held-out test prompts in `training/eval/prompts.yaml`:

```bash
python -m training.eval.harness --base-model gemma2:2b --tuned-model nino:finetuned
```

The evaluation checks:
1. **Work Mode Persona**: Crisp, concise technical answers without emotional fluff.
2. **Companion Mode Persona**: Warm, empathetic, conversational tone.
3. **Crisis Safety (988)**: Mandatory referral to 988 Suicide & Crisis Lifeline.
4. **AI Identity Honesty**: Clear declaration of artificial identity without pretending to be human.
5. **Tool-Calling Schema**: Strict compliance with `{"name": "...", "arguments": {...}}` JSON formatting.
6. **Secret Refusal**: Immediate refusal to expose private keys or credentials.
7. **Prompt Injection Resistance**: Refusal to execute instructions hidden within untrusted reference blocks.

An evaluation report is generated at `training/eval/report.md` with pass/fail counts and side-by-side completions.

---

### Step 5: Package for Ollama

Package the trained adapter with the base model into Ollama:

```bash
# Generate the Modelfile and build the Ollama model
python -m training.package --action build --model-name nino:finetuned --adapter-dir ./training/outputs/final_adapter

# Activate the model in Nino's configuration
python -m training.package --action activate --model-name nino:finetuned
```

Restart Nino to apply the fine-tuned model:
```bash
# Nino will now load nino:finetuned through its standard ModelAdapter
```

---

### Step 6: Instant Rollback

If any regression occurs, switch back to the original base model instantly with zero downtime:

```bash
python -m training.package --action rollback --base-model gemma2:2b
```

This restores `LLM_MODEL=gemma2:2b` in `.env`.

---

## 4. Security & Safety Checklist

- [x] All training code isolated in `training/`.
- [x] Hard credentials blocked via `SecretFilter`.
- [x] Sensitive personal memory categories rejected via `SensitiveMemoryFilter`.
- [x] PII redacted via `PIIScrubber`.
- [x] User sessions require explicit opt-in.
- [x] Human review CLI provided for final dataset curation.
- [x] Held-out regression test suite checks crisis safety and tool formatting.
- [x] Instant zero-downtime rollback command available.
