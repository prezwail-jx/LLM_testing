import argparse
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

# Mitigate OpenMP shared-memory init issues on some local macOS/conda setups.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


app = FastAPI(title="Transformers OpenAI-Compatible Server")
MODEL = None
TOKENIZER = None
MODEL_ID = ""
DEVICE = "cpu"


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_prompt(messages: List[ChatMessage]) -> str:
    simple_messages = [{"role": m.role, "content": m.content} for m in messages]
    if hasattr(TOKENIZER, "apply_chat_template"):
        try:
            return TOKENIZER.apply_chat_template(
                simple_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback when local jinja2/template stack is incompatible.
            pass
    return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"


def tensor_on_device(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v.to(DEVICE) for k, v in inputs.items()}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = build_prompt(req.messages)
    inputs = TOKENIZER(prompt, return_tensors="pt")
    inputs = tensor_on_device(inputs)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    do_sample = req.temperature > 0
    generation_kwargs = {
        **inputs,
        "max_new_tokens": req.max_tokens,
        "do_sample": do_sample,
        "temperature": max(req.temperature, 1e-6) if do_sample else 1.0,
        "pad_token_id": TOKENIZER.eos_token_id,
    }

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_ts = int(time.time())

    if not req.stream:
        with torch.inference_mode():
            output_ids = MODEL.generate(**generation_kwargs)
        new_ids = output_ids[0][prompt_tokens:]
        text = TOKENIZER.decode(new_ids, skip_special_tokens=True)
        completion_tokens = int(new_ids.shape[0])
        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return JSONResponse(payload)

    streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer

    def run_generate() -> None:
        with torch.inference_mode():
            MODEL.generate(**generation_kwargs)

    thread = threading.Thread(target=run_generate, daemon=True)
    thread.start()

    def event_stream():
        produced_text = ""
        for token_text in streamer:
            produced_text += token_text
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {JSONResponse(chunk).body.decode('utf-8')}\n\n"

        completion_tokens = len(TOKENIZER.encode(produced_text, add_special_tokens=False))
        done_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {JSONResponse(done_chunk).body.decode('utf-8')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    global MODEL, TOKENIZER, MODEL_ID, DEVICE

    args = parse_args()
    MODEL_ID = args.model_id
    DEVICE = select_device()

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    dtype = torch.float32
    if DEVICE in {"cuda", "mps"}:
        dtype = torch.float16

    MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    MODEL.to(DEVICE)
    MODEL.eval()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
