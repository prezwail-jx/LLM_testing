import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from openai import OpenAI


@dataclass
class RequestResult:
    prompt_id: int
    prompt_text: str
    backend: str
    model_id: str
    start_ts: str
    first_token_ts: Optional[str]
    end_ts: str
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    e2e_latency_ms: Optional[float]
    output_tokens: int
    throughput_tok_per_s: Optional[float]
    status: str
    error_message: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            prompts.append(text)
    return prompts


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_vals[low]
    frac = rank - low
    return sorted_vals[low] * (1.0 - frac) + sorted_vals[high] * frac


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: float("nan") for k in ["mean", "p50", "p90", "p95", "min", "max"]}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def estimate_tokens(text: str) -> int:
    # Lightweight fallback when usage is unavailable.
    return max(1, len(text.split())) if text.strip() else 0


def run_one(
    client: OpenAI,
    model_id: str,
    backend: str,
    prompt_id: int,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
) -> RequestResult:
    start_perf = time.perf_counter()
    start_ts = now_iso()
    first_token_perf: Optional[float] = None
    first_token_ts: Optional[str] = None
    output_chunks: List[str] = []
    completion_tokens: Optional[int] = None

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                completion_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                if first_token_perf is None:
                    first_token_perf = time.perf_counter()
                    first_token_ts = now_iso()
                output_chunks.append(content)

        end_perf = time.perf_counter()
        end_ts = now_iso()

        output_text = "".join(output_chunks)
        output_tokens = completion_tokens if completion_tokens is not None else estimate_tokens(output_text)

        e2e_s = max(0.0, end_perf - start_perf)
        ttft_s = max(0.0, first_token_perf - start_perf) if first_token_perf is not None else e2e_s
        gen_s = max(0.0, end_perf - first_token_perf) if first_token_perf is not None else 0.0

        tpot_ms: Optional[float]
        if output_tokens > 0 and first_token_perf is not None:
            tpot_ms = (gen_s * 1000.0) / output_tokens
        else:
            tpot_ms = None

        throughput = (output_tokens / e2e_s) if e2e_s > 0 else None

        return RequestResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            backend=backend,
            model_id=model_id,
            start_ts=start_ts,
            first_token_ts=first_token_ts,
            end_ts=end_ts,
            ttft_ms=ttft_s * 1000.0,
            tpot_ms=tpot_ms,
            e2e_latency_ms=e2e_s * 1000.0,
            output_tokens=int(output_tokens),
            throughput_tok_per_s=throughput,
            status="success",
            error_message="",
        )
    except Exception as exc:
        end_ts = now_iso()
        return RequestResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            backend=backend,
            model_id=model_id,
            start_ts=start_ts,
            first_token_ts=None,
            end_ts=end_ts,
            ttft_ms=None,
            tpot_ms=None,
            e2e_latency_ms=None,
            output_tokens=0,
            throughput_tok_per_s=None,
            status="error",
            error_message=str(exc),
        )


def to_dict(result: RequestResult) -> Dict[str, object]:
    return {
        "prompt_id": result.prompt_id,
        "prompt_text": result.prompt_text,
        "backend": result.backend,
        "model_id": result.model_id,
        "start_ts": result.start_ts,
        "first_token_ts": result.first_token_ts,
        "end_ts": result.end_ts,
        "ttft_ms": result.ttft_ms,
        "tpot_ms": result.tpot_ms,
        "e2e_latency_ms": result.e2e_latency_ms,
        "output_tokens": result.output_tokens,
        "throughput_tok_per_s": result.throughput_tok_per_s,
        "status": result.status,
        "error_message": result.error_message,
    }


def fmt(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.2f}"


def write_report(
    output_path: str,
    model_id: str,
    backend: str,
    prompts_total: int,
    results: List[RequestResult],
    run_start_ts: str,
    run_end_ts: str,
) -> None:
    success = [r for r in results if r.status == "success"]
    errors = [r for r in results if r.status != "success"]

    ttft_vals = [r.ttft_ms for r in success if r.ttft_ms is not None]
    tpot_vals = [r.tpot_ms for r in success if r.tpot_ms is not None]
    e2e_vals = [r.e2e_latency_ms for r in success if r.e2e_latency_ms is not None]
    tps_vals = [r.throughput_tok_per_s for r in success if r.throughput_tok_per_s is not None]

    ttft_summary = summarize(ttft_vals)
    tpot_summary = summarize(tpot_vals)
    e2e_summary = summarize(e2e_vals)
    tps_summary = summarize(tps_vals)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Model Benchmark Report\n")
        f.write("=" * 72 + "\n")
        f.write(f"Run start (UTC): {run_start_ts}\n")
        f.write(f"Run end   (UTC): {run_end_ts}\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Backend: {backend}\n")
        f.write(f"OS: {platform.platform()}\n")
        f.write(f"Python: {sys.version.split()[0]}\n")
        f.write(f"Total prompts: {prompts_total}\n")
        f.write(f"Success: {len(success)}\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write("\n")

        f.write("Metric summary (success only)\n")
        f.write("-" * 72 + "\n")
        f.write(
            "TTFT(ms): mean={mean} p50={p50} p90={p90} p95={p95} min={min} max={max}\n".format(
                **{k: fmt(v) for k, v in ttft_summary.items()}
            )
        )
        f.write(
            "TPOT(ms): mean={mean} p50={p50} p90={p90} p95={p95} min={min} max={max}\n".format(
                **{k: fmt(v) for k, v in tpot_summary.items()}
            )
        )
        f.write(
            "E2E(ms):  mean={mean} p50={p50} p90={p90} p95={p95} min={min} max={max}\n".format(
                **{k: fmt(v) for k, v in e2e_summary.items()}
            )
        )
        f.write(
            "TPS(tok/s): mean={mean} p50={p50} p90={p90} p95={p95} min={min} max={max}\n".format(
                **{k: fmt(v) for k, v in tps_summary.items()}
            )
        )
        f.write("\n")

        f.write("Per-prompt results\n")
        f.write("-" * 72 + "\n")
        for r in results:
            f.write(
                f"[{r.prompt_id:02d}] status={r.status} ttft_ms={fmt(r.ttft_ms)} "
                f"tpot_ms={fmt(r.tpot_ms)} e2e_ms={fmt(r.e2e_latency_ms)} "
                f"tokens={r.output_tokens} tps={fmt(r.throughput_tok_per_s)}\n"
            )
            if r.status != "success":
                f.write(f"     error={r.error_message}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout-s", type=float, default=120)
    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--min-success-rate", type=float, default=0.90)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = read_prompts(args.prompts_file)
    if not prompts:
        raise RuntimeError(f"No prompts found in {args.prompts_file}")

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.host}:{args.port}/v1",
        timeout=args.timeout_s,
    )

    run_start_ts = now_iso()

    warmup_n = max(0, min(args.warmup_requests, len(prompts)))
    for i in range(warmup_n):
        _ = run_one(
            client=client,
            model_id=args.model_id,
            backend=args.backend,
            prompt_id=-(i + 1),
            prompt_text=prompts[i],
            max_new_tokens=min(32, args.max_new_tokens),
            temperature=args.temperature,
        )

    results: List[RequestResult] = []
    for idx, prompt in enumerate(prompts, start=1):
        result = run_one(
            client=client,
            model_id=args.model_id,
            backend=args.backend,
            prompt_id=idx,
            prompt_text=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        results.append(result)

    run_end_ts = now_iso()

    raw_path = os.path.join(args.output_dir, "raw_results.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(to_dict(r), ensure_ascii=False) + "\n")

    meta = {
        "run_start_ts": run_start_ts,
        "run_end_ts": run_end_ts,
        "model_id": args.model_id,
        "backend": args.backend,
        "host": args.host,
        "port": args.port,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "timeout_s": args.timeout_s,
        "warmup_requests": warmup_n,
        "prompts_file": args.prompts_file,
        "platform": platform.platform(),
        "python_version": sys.version,
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    write_report(
        output_path=os.path.join(args.output_dir, "report.txt"),
        model_id=args.model_id,
        backend=args.backend,
        prompts_total=len(prompts),
        results=results,
        run_start_ts=run_start_ts,
        run_end_ts=run_end_ts,
    )

    success_cnt = sum(1 for r in results if r.status == "success")
    success_rate = success_cnt / len(results)
    print(f"success_rate={success_rate:.2%} ({success_cnt}/{len(results)})")

    if success_rate < args.min_success_rate:
        raise SystemExit(
            f"Success rate {success_rate:.2%} is lower than threshold {args.min_success_rate:.2%}"
        )


if __name__ == "__main__":
    main()
