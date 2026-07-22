#!/usr/bin/env python3
"""Build and validate the scheduled 40-request PD-flip experiment trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_forced_token(tokenizer: Any, forced_text: str) -> int:
    token_ids = tokenizer.encode(forced_text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError("forced_text must encode to exactly one token")
    if tokenizer.decode(token_ids) != forced_text:
        raise ValueError(
            "forced_text must decode from its token to exactly the same text"
        )
    return int(token_ids[0])


def apply_output_contract(
    row: dict[str, Any],
    *,
    max_tokens: int,
    forced_text: str,
    forced_token_id: int,
    custom_logit_processor: str,
    model: str | None = None,
) -> None:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not isinstance(forced_text, str) or not forced_text:
        raise ValueError("forced_text must be nonempty")
    if not isinstance(forced_token_id, int) or forced_token_id < 0:
        raise ValueError("forced_token_id must be a non-negative integer")
    if not isinstance(custom_logit_processor, str) or not custom_logit_processor:
        raise ValueError("custom_logit_processor must be nonempty")

    body = row.setdefault("body", {})
    if model is not None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be nonempty")
        row["model"] = model
        body["model"] = model
    custom_params = body.setdefault("custom_params", {})
    row["max_tokens"] = max_tokens
    row["stream"] = True
    body["max_tokens"] = max_tokens
    body["temperature"] = 0.0
    body["stream"] = True
    body["ignore_eos"] = True
    body["stop"] = None
    body["stream_options"] = {"include_usage": True}
    body["custom_logit_processor"] = custom_logit_processor
    custom_params["forced_text"] = forced_text
    custom_params["forced_token_id"] = forced_token_id


def apply_natural_output_contract(
    row: dict[str, Any], *, max_tokens: int, model: str | None = None
) -> None:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    body = row.setdefault("body", {})
    if model is not None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be nonempty")
        row["model"] = model
        body["model"] = model
    row["max_tokens"] = max_tokens
    row["stream"] = True
    body["max_tokens"] = max_tokens
    body["temperature"] = 0.0
    body["stream"] = True
    body["ignore_eos"] = True
    body["stop"] = None
    body["stream_options"] = {"include_usage": True}
    body.pop("custom_logit_processor", None)
    custom_params = body.setdefault("custom_params", {})
    custom_params.pop("forced_text", None)
    custom_params.pop("forced_token_id", None)


def _validate_trace(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 40:
        raise ValueError(f"expected 40 requests, got {len(rows)}")
    request_ids = [row.get("request_id") for row in rows]
    if any(
        not isinstance(request_id, str) or not request_id for request_id in request_ids
    ):
        raise ValueError("every request needs a nonempty request_id")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_id values must be unique")
    prompts = []
    for row in rows:
        messages = (row.get("body") or {}).get("messages") or []
        content = messages[0].get("content") if messages else None
        if not isinstance(content, str) or not content:
            raise ValueError("every request needs a nonempty Prompt")
        prompts.append(content)
    if len(set(prompts)) != len(prompts):
        raise ValueError("every request Prompt must be unique")
    expected_kinds = ["long", "short"] * 20
    if [row.get("prompt_kind") for row in rows] != expected_kinds:
        raise ValueError("requests must alternate long, short")
    if sum(row.get("prompt_chars") == 10000 for row in rows) != 20:
        raise ValueError("expected 20 requests with prompt_chars=10000")
    if sum(row.get("prompt_chars") == 1000 for row in rows) != 20:
        raise ValueError("expected 20 requests with prompt_chars=1000")
    if not all(
        float(row.get("ttft_slo_s", 0)) > 0 and float(row.get("tpot_slo_s", 0)) > 0
        for row in rows
    ):
        raise ValueError("every request needs positive TTFT and TPOT SLO values")


def _validate_output_contract(
    rows: list[dict[str, Any]],
    *,
    max_tokens: int,
    forced_text: str,
    forced_token_id: int,
    custom_logit_processor: str,
    model: str | None = None,
) -> None:
    for index, row in enumerate(rows):
        body = row.get("body") or {}
        custom_params = body.get("custom_params") or {}
        if model is not None and (
            row.get("model") != model or body.get("model") != model
        ):
            raise ValueError(f"request {index} model contract mismatch")
        if row.get("max_tokens") != max_tokens or body.get("max_tokens") != max_tokens:
            raise ValueError(f"request {index} max_tokens contract mismatch")
        if row.get("stream") is not True or body.get("stream") is not True:
            raise ValueError(f"request {index} must use streaming")
        if body.get("temperature") != 0.0:
            raise ValueError(f"request {index} temperature contract mismatch")
        if body.get("ignore_eos") is not True or body.get("stop") is not None:
            raise ValueError(f"request {index} EOS/stop contract mismatch")
        if body.get("stream_options") != {"include_usage": True}:
            raise ValueError(f"request {index} must request streaming usage")
        if body.get("custom_logit_processor") != custom_logit_processor:
            raise ValueError(f"request {index} custom_logit_processor mismatch")
        if custom_params.get("forced_text") != forced_text:
            raise ValueError(f"request {index} forced_text mismatch")
        if custom_params.get("forced_token_id") != forced_token_id:
            raise ValueError(f"request {index} forced_token_id mismatch")


def _validate_natural_output_contract(
    rows: list[dict[str, Any]], *, max_tokens: int, model: str | None = None
) -> None:
    for index, row in enumerate(rows):
        body = row.get("body") or {}
        custom_params = body.get("custom_params") or {}
        if model is not None and (
            row.get("model") != model or body.get("model") != model
        ):
            raise ValueError(f"request {index} model contract mismatch")
        if row.get("max_tokens") != max_tokens or body.get("max_tokens") != max_tokens:
            raise ValueError(f"request {index} max_tokens contract mismatch")
        if row.get("stream") is not True or body.get("stream") is not True:
            raise ValueError(f"request {index} must use streaming")
        if body.get("temperature") != 0.0:
            raise ValueError(f"request {index} temperature contract mismatch")
        if body.get("ignore_eos") is not True or body.get("stop") is not None:
            raise ValueError(f"request {index} EOS/stop contract mismatch")
        if body.get("stream_options") != {"include_usage": True}:
            raise ValueError(f"request {index} must request streaming usage")
        if "custom_logit_processor" in body:
            raise ValueError(f"request {index} must not use custom_logit_processor")
        if "forced_text" in custom_params or "forced_token_id" in custom_params:
            raise ValueError(f"request {index} must not use forced sampling fields")


def prepare_trace(
    source: Path,
    output: Path,
    manifest: Path,
    wave_size: int,
    wave_gap_seconds: float,
    intra_wave_interval_seconds: float,
    ttft_slo_override_seconds: float = 0.0,
    long_ttft_slo_seconds: float = 0.0,
    short_ttft_slo_seconds: float = 0.0,
    tpot_slo_override_seconds: float = 0.0,
    max_tokens: int | None = None,
    forced_text: str | None = None,
    forced_token_id: int | None = None,
    custom_logit_processor: str | None = None,
    model: str | None = None,
    natural_output: bool = False,
) -> None:
    if wave_size <= 0:
        raise ValueError("wave_size must be positive")
    if wave_gap_seconds < 0 or intra_wave_interval_seconds < 0:
        raise ValueError("trace timing values must be non-negative")
    if any(
        value < 0
        for value in (
            ttft_slo_override_seconds,
            long_ttft_slo_seconds,
            short_ttft_slo_seconds,
            tpot_slo_override_seconds,
        )
    ):
        raise ValueError("SLO override values must be non-negative")
    if ttft_slo_override_seconds > 0 and (
        long_ttft_slo_seconds > 0 or short_ttft_slo_seconds > 0
    ):
        raise ValueError(
            "use either the uniform TTFT override or the long/short overrides"
        )
    if (long_ttft_slo_seconds > 0) != (short_ttft_slo_seconds > 0):
        raise ValueError("long and short TTFT SLO overrides must be set together")

    source_rows = _load_jsonl(source)
    _validate_trace(source_rows)
    scheduled_rows = []
    for index, row in enumerate(source_rows):
        scheduled = dict(row)
        scheduled["arrival_offset_s"] = round(
            (index // wave_size) * wave_gap_seconds
            + (index % wave_size) * intra_wave_interval_seconds,
            9,
        )
        effective_ttft_slo = ttft_slo_override_seconds
        if long_ttft_slo_seconds > 0:
            effective_ttft_slo = (
                long_ttft_slo_seconds
                if scheduled.get("prompt_kind") == "long"
                else short_ttft_slo_seconds
            )
        if effective_ttft_slo > 0:
            scheduled["ttft_slo_s"] = effective_ttft_slo
            (
                scheduled.setdefault("body", {})
                .setdefault("custom_params", {})
                .setdefault("pd_flip_slo", {})["ttft_seconds"]
            ) = effective_ttft_slo
        if tpot_slo_override_seconds > 0:
            scheduled["tpot_slo_s"] = tpot_slo_override_seconds
            (
                scheduled.setdefault("body", {})
                .setdefault("custom_params", {})
                .setdefault("pd_flip_slo", {})["tpot_seconds"]
            ) = tpot_slo_override_seconds
        if max_tokens is not None:
            if natural_output:
                apply_natural_output_contract(
                    scheduled, max_tokens=max_tokens, model=model
                )
            else:
                apply_output_contract(
                    scheduled,
                    max_tokens=max_tokens,
                    forced_text=forced_text or "",
                    forced_token_id=(
                        forced_token_id if forced_token_id is not None else -1
                    ),
                    custom_logit_processor=custom_logit_processor or "",
                    model=model,
                )
        scheduled_rows.append(scheduled)

    if any(
        left["arrival_offset_s"] > right["arrival_offset_s"]
        for left, right in zip(scheduled_rows, scheduled_rows[1:])
    ):
        raise ValueError(
            "wave_gap_seconds is too small to keep arrival offsets monotonic"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in scheduled_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    reloaded = _load_jsonl(output)
    _validate_trace(reloaded)
    if max_tokens is not None:
        if natural_output:
            _validate_natural_output_contract(
                reloaded, max_tokens=max_tokens, model=model
            )
        else:
            _validate_output_contract(
                reloaded,
                max_tokens=max_tokens,
                forced_text=forced_text or "",
                forced_token_id=(
                    forced_token_id if forced_token_id is not None else -1
                ),
                custom_logit_processor=custom_logit_processor or "",
                model=model,
            )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "source_trace": str(source),
                "effective_trace": str(output),
                "source_sha256": _sha256(source),
                "effective_sha256": _sha256(output),
                "request_count": len(reloaded),
                "wave_size": wave_size,
                "wave_gap_seconds": wave_gap_seconds,
                "intra_wave_interval_seconds": intra_wave_interval_seconds,
                "ttft_slo_override_seconds": ttft_slo_override_seconds,
                "long_ttft_slo_seconds": long_ttft_slo_seconds,
                "short_ttft_slo_seconds": short_ttft_slo_seconds,
                "tpot_slo_override_seconds": tpot_slo_override_seconds,
                "last_arrival_offset_s": reloaded[-1]["arrival_offset_s"],
                "max_tokens": max_tokens,
                "forced_text": forced_text,
                "forced_token_id": forced_token_id,
                "output_contract": "natural" if natural_output else "forced",
                "model": model,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wave-size", type=int, required=True)
    parser.add_argument("--wave-gap-seconds", type=float, required=True)
    parser.add_argument("--intra-wave-interval-seconds", type=float, required=True)
    parser.add_argument("--ttft-slo-override-seconds", type=float, default=0.0)
    parser.add_argument("--long-ttft-slo-seconds", type=float, default=0.0)
    parser.add_argument("--short-ttft-slo-seconds", type=float, default=0.0)
    parser.add_argument("--tpot-slo-override-seconds", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--forced-text")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--natural-output", action="store_true")
    parser.add_argument("--model", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.natural_output:
        prepare_trace(
            source=args.source,
            output=args.output,
            manifest=args.manifest,
            wave_size=args.wave_size,
            wave_gap_seconds=args.wave_gap_seconds,
            intra_wave_interval_seconds=args.intra_wave_interval_seconds,
            ttft_slo_override_seconds=args.ttft_slo_override_seconds,
            long_ttft_slo_seconds=args.long_ttft_slo_seconds,
            short_ttft_slo_seconds=args.short_ttft_slo_seconds,
            tpot_slo_override_seconds=args.tpot_slo_override_seconds,
            max_tokens=args.max_tokens,
            model=args.model,
            natural_output=True,
        )
        return

    if args.forced_text is None or args.tokenizer_path is None:
        parser.error("forced mode requires --forced-text and --tokenizer-path")

    from transformers import AutoTokenizer

    from sglang.srt.sampling.custom_logit_processor import (
        ForcedSingleTokenLogitProcessor,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    forced_token_id = resolve_forced_token(tokenizer, args.forced_text)
    custom_logit_processor = ForcedSingleTokenLogitProcessor.to_str()
    prepare_trace(
        source=args.source,
        output=args.output,
        manifest=args.manifest,
        wave_size=args.wave_size,
        wave_gap_seconds=args.wave_gap_seconds,
        intra_wave_interval_seconds=args.intra_wave_interval_seconds,
        ttft_slo_override_seconds=args.ttft_slo_override_seconds,
        long_ttft_slo_seconds=args.long_ttft_slo_seconds,
        short_ttft_slo_seconds=args.short_ttft_slo_seconds,
        tpot_slo_override_seconds=args.tpot_slo_override_seconds,
        max_tokens=args.max_tokens,
        forced_text=args.forced_text,
        forced_token_id=forced_token_id,
        custom_logit_processor=custom_logit_processor,
        model=args.model,
    )


if __name__ == "__main__":
    main()
