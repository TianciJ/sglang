#!/usr/bin/env python3
"""Build the frozen Qwen3-Next 80B trace used by the PD Flip A/B run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.playground.disaggregation.pd_flip_prepare_trace import (
    apply_output_contract,
)


_PROMPT_TEXT = (
    "在分布式推理系统中，请持续分析请求调度、缓存管理、网络传输与计算资源之间的关系。"
    "重点描述每个阶段的输入输出、状态变化、等待原因和可观测指标，并保持内容连贯。"
    "本段仅用于构造稳定的性能测试负载，不要求回答问题，也不要省略上下文。"
)


def build_prompt(*, request_index: int, prompt_kind: str, run_nonce: str) -> str:
    if prompt_kind not in {"short", "long"}:
        raise ValueError(f"unknown prompt_kind: {prompt_kind}")
    if not run_nonce:
        raise ValueError("run_nonce must be nonempty")
    if not 0 <= request_index < 40:
        raise ValueError("request_index must be in [0, 40)")

    target_chars = 1_000 if prompt_kind == "short" else 10_000
    unique_first_character = chr(0x4E00 + request_index)
    nonce = f"{unique_first_character}|{run_nonce}:req-{request_index:02d}:"
    request_marker = f"请求编号{request_index:02d}，负载类型{prompt_kind}。"
    seed = nonce + request_marker + _PROMPT_TEXT
    repetitions = (target_chars + len(seed) - 1) // len(seed)
    return (seed * repetitions)[:target_chars]


def build_qwen80b_trace(
    *,
    run_nonce: str,
    model: str,
    forced_token_id: int,
    forced_text: str,
    custom_logit_processor: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(40):
        prompt_kind = "long" if index % 2 == 0 else "short"
        user_content = build_prompt(
            request_index=index,
            prompt_kind=prompt_kind,
            run_nonce=run_nonce,
        )
        ttft_slo_s = 5.0 if prompt_kind == "long" else 2.0
        row: dict[str, Any] = {
            "request_id": f"qwen80b-{index:02d}",
            "prompt_kind": prompt_kind,
            "prompt_chars": len(user_content),
            "user_content": user_content,
            "ttft_slo_s": ttft_slo_s,
            "tpot_slo_s": 0.05,
            "arrival_offset_s": (index // 10) * 7.5 + (index % 10) * 0.5,
            "body": {
                "messages": [{"role": "user", "content": user_content}],
                "custom_params": {
                    "pd_flip_slo": {
                        "ttft_seconds": ttft_slo_s,
                        "tpot_seconds": 0.05,
                    }
                },
            },
        }
        apply_output_contract(
            row,
            max_tokens=10_000,
            forced_text=forced_text,
            forced_token_id=forced_token_id,
            custom_logit_processor=custom_logit_processor,
            model=model,
        )
        rows.append(row)
    return rows


def write_trace(
    trace: Sequence[dict[str, Any]], output: Path, manifest: Path
) -> dict[str, Any]:
    if len(trace) != 40:
        raise ValueError(f"expected 40 requests, got {len(trace)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in trace:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )

    trace_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    result = {
        "trace_path": str(output),
        "trace_sha256": trace_sha256,
        "request_count": len(trace),
        "short_requests": sum(row.get("prompt_kind") == "short" for row in trace),
        "long_requests": sum(row.get("prompt_kind") == "long" for row in trace),
        "last_arrival_offset_s": float(trace[-1]["arrival_offset_s"]),
        "model": trace[0].get("model"),
        "max_tokens": trace[0].get("max_tokens"),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--forced-token-id", required=True, type=int)
    parser.add_argument("--forced-text", required=True)
    parser.add_argument("--custom-logit-processor", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trace = build_qwen80b_trace(
        run_nonce=args.run_nonce,
        model=args.model,
        forced_token_id=args.forced_token_id,
        forced_text=args.forced_text,
        custom_logit_processor=args.custom_logit_processor,
    )
    write_trace(trace, args.output, args.manifest)


if __name__ == "__main__":
    main()
