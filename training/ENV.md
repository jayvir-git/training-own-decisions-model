# Training environment (2026-10-28)

Native Windows 11, no WSL. Venv at `.venv/` (gitignored).

## Hardware
- NVIDIA GeForce RTX 3050 4GB Laptop GPU (Ampere, bf16 supported), driver 592.00 (CUDA 13.1 runtime per `nvidia-smi`), WDDM.
- No CUDA toolkit / `nvcc` installed; not needed (wheels bundle CUDA 13.0 runtime; Triton for Windows JIT works without it).

## Python
- The `py` launcher lists 3.12/3.13 under `%USERPROFILE%\scoop\apps\python\…`, but that directory no longer exists (stale registry entries). Only 3.14 is really installed.
- Unsloth's Windows docs require 3.10–3.12, so: `python -m pip install --user uv` (uv 0.13.6) → `uv python install 3.12` → CPython **3.12.16** (uv-managed) → `uv venv .venv --python 3.12`.

## Install commands (per unsloth.ai/docs Windows install, Method "pip after torch")
```
python -m uv pip install --python .venv/Scripts/python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -m uv pip install --python .venv/Scripts/python.exe unsloth --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```
`unsloth` pulls `triton-windows` itself; no manual Triton/Visual Studio step was needed.

## Versions
| package | version |
|---|---|
| torch | 2.14.1+cu130 (CUDA 13.0) |
| torchvision | 0.29.1+cu130 |
| torchaudio | 2.11.0+cu130 (resolver picked an older build; unused) |
| triton-windows | 3.9.0.post30 |
| unsloth / unsloth-zoo | 2026.10.9 / 2026.10.9 |
| transformers | 5.17.0 |
| peft | 0.21.2 |
| trl | 1.15.0 |
| accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| xformers | 0.0.35 |
| datasets | 4.8.5 |
| tokenizers | 0.23.3 |
| huggingface-hub | 1.33.0 |
| numpy | 2.5.2 |

`torch.cuda.is_available() == True`, device name reported correctly, `is_bf16_supported() == True`.

## Notes
- `import unsloth` takes ~80 s warm (first import > 5 min while caches/kernels build). Not a hang.
- Decision API verified in `.venv/Lib/site-packages/unsloth/models/decision.py` (AGPL-3.0-only header on the decision-model files; core Unsloth is Apache-2.0):
  `FastDecisionModel.from_pretrained(model_name, max_seq_length, load_in_4bit, …)` — a plain LM such as `unsloth/Qwen3.5-0.8B` gets a fresh "clef" JointSchemaHead (width 512 for hidden 1024);
  `get_peft_model(model, r=64, lora_alpha=64, …)` defaults — pass r/alpha explicitly;
  `build_dataset(rows, tokenizer, model) -> (items, report)`; `split_holdout(items, seed, fraction=0.1)`;
  `DecisionTrainer(model, args, head_learning_rate=None→1e-4, …)`; `calibrate`, `evaluate`, `predict(model, tokenizer, state, questions)`;
  `model.save_pretrained` (LoRA + head) / `model.save_pretrained_merged` (16-bit).
