# LLM Testing Automation

Automated local benchmark pipeline for small LLMs.
It starts a local OpenAI-compatible model server, runs prompt tests, and writes a TXT/JSON report.

## 1. Prerequisites

- OS: macOS or Linux
- Python: 3.9 to 3.11
- Git
- Network access to Hugging Face model hub (first run downloads model files)

## 2. Setup On A New Computer

1. If you use Conda, exit `base` first (important):

```bash
conda deactivate  #outside of venv
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Confirm Python is from this project:

```bash
which python
python -V
```

Expected `python` path:
`.../LLM_testing/.venv/bin/python`

4. Install dependencies:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

## 3. Configure Benchmark (Optional)

Edit `benchmarks/config.env` only if you want persistent defaults:

```bash
vim benchmarks/config.env
```

Prompt set options:

- `benchmarks/prompts_dev.txt`: short and simple prompts for fast script debugging.
- `benchmarks/prompts_eval.txt`: larger and more diverse prompts for full evaluation.
- `benchmarks/prompts_default.txt`: default prompt file (balanced set).

Switch by setting `PROMPTS_FILE` in `benchmarks/config.env`, for example:

```env
PROMPTS_FILE=benchmarks/prompts_dev.txt
```

### Config Reference

`benchmarks/config.env` supports the following keys:

- `MODEL_ID`: Hugging Face model id to load.
  - Default: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
  - Example: `Qwen/Qwen2.5-0.5B-Instruct`
- `BACKEND`: Inference backend selection.
  - Values: `auto`, `transformers`, `vllm`
  - Default: `transformers`
- `ENABLE_FALLBACK`: Whether to fallback from vLLM to transformers when vLLM fails to start.
  - Values: `1` (enabled), `0` (disabled)
  - Default: `1`
- `HOST`: Server bind host.
  - Default: `127.0.0.1`
- `PORT`: Server bind port.
  - Default: `8000`
- `MAX_NEW_TOKENS`: Max generated tokens per request.
  - Default: `128`
- `TEMPERATURE`: Sampling temperature.
  - Default: `0.0`
- `TIMEOUT_S`: Startup health-check timeout and request timeout in seconds.
  - Default: `900` in script fallback (can be set higher in config)
- `WARMUP_REQUESTS`: Number of warmup requests before measured run.
  - Default: `1`
- `MIN_SUCCESS_RATE`: Minimum success ratio required by evaluator.
  - Default: `0.90`
- `OUTPUT_DIR`: Output root directory for benchmark artifacts.
  - Default: `outputs`
- `PROMPTS_FILE`: Prompt file path (relative to repo root or absolute path).
  - Default: `benchmarks/prompts_default.txt`
- `PRELOAD_MODEL`: Whether to pre-download model before server startup.
  - Values: `1` (download first), `0` (download on demand)
  - Default: `0`
- `MODEL_DIR`: Target directory used by `scripts/download_model.sh` for local model download.
  - Default: `model`

## 4. Run

```bash
bash scripts/run_benchmark.sh
```

Override model and prompt set from command line:

```bash
bash scripts/run_benchmark.sh \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --prompts-file benchmarks/prompts_dev.txt
```

The script prints runtime Python at startup:

- `[env] python=...`

Use that line to verify you are running inside `.venv`.

## 5. Outputs

Each run creates `outputs/<timestamp>/` with:

- `report.txt`: summary report
- `raw_results.jsonl`: per-prompt raw records
- `run_meta.json`: run metadata and pass/fail status
- `server_*.log`: backend startup/runtime log

### Metric Definitions

- `TTFT(ms)` (Time To First Token): time from request start to first generated token.
- `TPOT(ms)` (Time Per Output Token): average generation time per output token after first token.
- `E2E(ms)` (End-to-End): total time from request start to request finish.
- `TPS(tok/s)` (Tokens Per Second): output token throughput over the full request.

Summary columns in `report.txt`:

- `mean`: arithmetic average.
- `p50`: median (50th percentile).
- `p90`: 90th percentile.
- `p95`: 95th percentile.
- `min`: fastest observed value.
- `max`: slowest observed value.

## 6. Optional Pre-Download

```bash
bash scripts/download_model.sh
```

Download a specific model from command line

```bash
bash scripts/download_model.sh --model-id <model_id>
```

## 7. Troubleshooting

1. Error: `local torch runtime is broken (python cannot import torch)`

- Usually wrong environment (Conda `base` mixed with `.venv`) or broken venv packages.
- Fix:

```bash
#outside of venv
conda deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

2. Error: `No module named 'vllm'`

- You selected `BACKEND=vllm` without installing vLLM.
- Fix: set `BACKEND=transformers`, or install vLLM on Linux+CUDA.

3. Error: `unknown backend`

- `BACKEND` value is misspelled.
- Valid values are exactly: `auto`, `vllm`, `transformers`.

## 8. Reproducibility Notes

- Keep dependencies from `requirements.txt` unchanged for consistent results.
- Use the same `MODEL_ID`, prompt file, and config values when comparing machines.
- Run only one benchmark process per machine/GPU at a time.
