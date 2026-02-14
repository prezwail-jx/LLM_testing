# LLM Testing Automation

Automated small-model benchmark pipeline with Hugging Face download, service startup, and TXT report output.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Edit config if needed:

```bash
vim benchmarks/config.env
```

3. Run benchmark:

```bash
bash scripts/run_benchmark.sh
```

4. See outputs:

- `outputs/<timestamp>/report.txt`
- `outputs/<timestamp>/raw_results.jsonl`
- `outputs/<timestamp>/run_meta.json`

## Notes

- Default model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Backend strategy:
  - `BACKEND=auto`: Linux+CUDA prefers vLLM, others use Transformers.
  - `BACKEND=vllm`: force vLLM (with optional fallback if enabled).
  - `BACKEND=transformers`: force local Transformers server.
- Mac Apple Silicon can run Transformers path directly. vLLM support on macOS is experimental.

## Optional pre-download

```bash
bash scripts/download_model.sh
```
