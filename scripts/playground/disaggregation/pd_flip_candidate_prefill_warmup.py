#!/usr/bin/env python3
"""Warm every candidate Prefill worker before a measured PD Flip run."""

import argparse
import copy
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


JsonDict = Dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class NodeSpec:
    name: str
    worker_url: str
    router_worker_id: str
    bootstrap_port: int


def parse_node_spec(raw: str) -> NodeSpec:
    fields: Dict[str, str] = {}
    for item in raw.split(","):
        key, separator, value = item.partition("=")
        if not separator or not key or key in fields:
            raise ValueError("invalid or duplicate node field: {}".format(item))
        fields[key] = value
    required = {"name", "worker_url", "router_worker_id", "bootstrap_port"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError("missing node fields: {}".format(", ".join(missing)))
    unknown = sorted(set(fields) - required)
    if unknown:
        raise ValueError("unknown node fields: {}".format(", ".join(unknown)))
    return NodeSpec(
        name=fields["name"],
        worker_url=fields["worker_url"].rstrip("/"),
        router_worker_id=fields["router_worker_id"],
        bootstrap_port=int(fields["bootstrap_port"]),
    )


def select_warmup_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, JsonDict]:
    selected: Dict[str, JsonDict] = {}
    for prompt_kind in ("long", "short"):
        row = next((item for item in rows if item.get("prompt_kind") == prompt_kind), None)
        if row is None:
            raise ValueError("missing warmup trace kind: {}".format(prompt_kind))
        copied = copy.deepcopy(dict(row))
        body = copy.deepcopy(dict(copied.get("body") or {}))
        if not body:
            raise ValueError("warmup trace kind {} has no request body".format(prompt_kind))
        body.pop("custom_params", None)
        body.pop("custom_logit_processor", None)
        body["max_tokens"] = 1
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        copied["body"] = body
        selected[prompt_kind] = copied
    return selected


def _response_items(response: Any) -> List[Mapping[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, Mapping)]
    if isinstance(response, Mapping):
        return [response]
    raise ValueError("worker role response is not an object or list")


def validate_worker_role(response: Any, role: str, *, require_idle: bool) -> None:
    items = _response_items(response)
    if not items:
        raise ValueError("worker role response has no shard status")
    for index, item in enumerate(items):
        if item.get("success") is not True:
            raise ValueError("worker shard {} did not report success".format(index))
        status = item.get("status") if isinstance(item.get("status"), Mapping) else item
        if status.get("role") != role:
            raise ValueError(
                "worker shard {} role is {}, expected {}".format(
                    index, status.get("role"), role
                )
            )
        if status.get("active_event_loop_role") != role:
            raise ValueError(
                "worker shard {} active event loop is {}, expected {}".format(
                    index, status.get("active_event_loop_role"), role
                )
            )
        if require_idle:
            if status.get("is_idle") is not True:
                raise ValueError("worker shard {} is not idle".format(index))
            if status.get("running_requests") or status.get("waiting_requests"):
                raise ValueError("worker shard {} still has requests".format(index))


def validate_router_topology(
    response: Any, *, expected_prefill: int, expected_decode: int
) -> None:
    if not isinstance(response, Mapping) or not isinstance(response.get("workers"), list):
        raise ValueError("router topology response has no workers list")
    workers = [item for item in response["workers"] if isinstance(item, Mapping)]
    if len(workers) != expected_prefill + expected_decode:
        raise ValueError("router topology has unexpected worker count")
    if any(bool(item.get("draining")) for item in workers):
        raise ValueError("router topology still has draining workers")
    roles = [str(item.get("effective_role") or item.get("role") or "").lower() for item in workers]
    if roles.count("prefill") != expected_prefill or roles.count("decode") != expected_decode:
        raise ValueError(
            "router topology is {}P{}D, expected {}P{}D".format(
                roles.count("prefill"),
                roles.count("decode"),
                expected_prefill,
                expected_decode,
            )
        )


def _router_workers(response: Any) -> List[Mapping[str, Any]]:
    if not isinstance(response, Mapping) or not isinstance(response.get("workers"), list):
        raise ValueError("router topology response has no workers list")
    return [item for item in response["workers"] if isinstance(item, Mapping)]


def _router_role(worker: Mapping[str, Any]) -> str:
    return str(worker.get("effective_role") or worker.get("role") or "").lower()


def validate_router_warmup_target(response: Any, *, target_worker_id: str) -> None:
    workers = _router_workers(response)
    routable_prefill = [
        item
        for item in workers
        if _router_role(item) == "prefill" and not bool(item.get("draining"))
    ]
    if len(routable_prefill) != 1:
        raise ValueError(
            "router has {} routable Prefill workers, expected one".format(
                len(routable_prefill)
            )
        )
    if str(routable_prefill[0].get("worker_id")) != target_worker_id:
        raise ValueError(
            "router routable Prefill is {}, expected {}".format(
                routable_prefill[0].get("worker_id"), target_worker_id
            )
        )
    routable_decode = [
        item
        for item in workers
        if _router_role(item) == "decode" and not bool(item.get("draining"))
    ]
    if len(routable_decode) < 2:
        raise ValueError(
            "router has {} routable Decode workers, expected at least two".format(
                len(routable_decode)
            )
        )


def validate_worker_fsm_safe(response: Any) -> None:
    if not isinstance(response, Mapping) or not isinstance(
        response.get("internal_states"), list
    ):
        raise ValueError("server info has no internal_states list")
    states = [item for item in response["internal_states"] if isinstance(item, Mapping)]
    if not states:
        raise ValueError("server info has no scheduler internal state")
    for index, state in enumerate(states):
        pd_flip = state.get("pd_flip")
        if not isinstance(pd_flip, Mapping) or pd_flip.get("enabled") is not True:
            raise ValueError("worker shard {} PD Flip FSM is not enabled".format(index))
        if pd_flip.get("state") != "safe" or pd_flip.get("direction") != "none":
            raise ValueError(
                "worker shard {} PD Flip FSM is not safe: state={} direction={}".format(
                    index, pd_flip.get("state"), pd_flip.get("direction")
                )
            )


class JsonlJournal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        row = {"timestamp_utc": utc_now(), "event": event}
        row.update(fields)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class HttpClient:
    def __init__(self, api_key: str, timeout_seconds: float):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _request(
        self,
        base_url: str,
        path: str,
        *,
        method: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> bytes:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "{} {} returned HTTP {}: {}".format(method, path, exc.code, body)
            ) from exc

    def get_json(self, base_url: str, path: str) -> Any:
        return json.loads(self._request(base_url, path, method="GET").decode("utf-8"))

    def post_json(self, base_url: str, path: str, payload: Mapping[str, Any]) -> Any:
        body = self._request(base_url, path, method="POST", payload=payload)
        return json.loads(body.decode("utf-8")) if body.strip() else {}

    def post_text(self, base_url: str, path: str) -> str:
        return self._request(base_url, path, method="POST").decode(
            "utf-8", errors="replace"
        )

    def stream_chat(self, router_url: str, body: Mapping[str, Any]) -> JsonDict:
        request = urllib.request.Request(
            router_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
        )
        started_utc = utc_now()
        started = time.monotonic()
        first_output_utc: Optional[str] = None
        first_output: Optional[float] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        finish_reason: Optional[str] = None
        status: Optional[int] = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = response.status
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    usage = event.get("usage") or {}
                    if usage.get("prompt_tokens") is not None:
                        prompt_tokens = int(usage["prompt_tokens"])
                    if usage.get("completion_tokens") is not None:
                        completion_tokens = int(usage["completion_tokens"])
                    choices = event.get("choices") or []
                    if choices:
                        finish_reason = choices[0].get("finish_reason") or finish_reason
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content") or delta.get("reasoning_content") or ""
                        if content and first_output is None:
                            first_output = time.monotonic()
                            first_output_utc = utc_now()
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "warmup returned HTTP {}: {}".format(exc.code, response_body)
            ) from exc
        finished = time.monotonic()
        finished_utc = utc_now()
        if status != 200:
            raise ValueError("warmup response status is {}".format(status))
        if first_output is None or first_output_utc is None:
            raise ValueError("warmup produced no non-empty first output")
        if completion_tokens != 1:
            raise ValueError(
                "warmup completion token count is {}, expected 1".format(completion_tokens)
            )
        if prompt_tokens is None:
            raise ValueError("warmup response has no prompt token usage")
        if finish_reason != "length":
            raise ValueError("warmup finish reason is {}".format(finish_reason))
        return {
            "response_status": status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "started_utc": started_utc,
            "first_output_utc": first_output_utc,
            "finished_utc": finished_utc,
            "ttft_s": first_output - started,
            "total_duration_s": finished - started,
        }


class CandidatePrefillWarmup:
    def __init__(
        self,
        *,
        router_url: str,
        nodes: Sequence[NodeSpec],
        initial_prefill_name: str,
        candidate_names: Sequence[str],
        selected_rows: Mapping[str, JsonDict],
        output_dir: Path,
        client: HttpClient,
        role_timeout_seconds: float,
        role_poll_seconds: float,
    ):
        self.router_url = router_url.rstrip("/")
        self.nodes = {node.name: node for node in nodes}
        if len(self.nodes) < 3:
            raise ValueError("candidate Prefill warmup requires at least three workers")
        self.expected_prefill = 1
        self.expected_decode = len(self.nodes) - 1
        self.expected_topology = "{}P{}D".format(
            self.expected_prefill, self.expected_decode
        )
        self.initial_prefill = self.nodes[initial_prefill_name]
        self.candidates = [self.nodes[name] for name in candidate_names]
        self.selected_rows = selected_rows
        self.output_dir = Path(output_dir)
        self.client = client
        self.role_timeout_seconds = role_timeout_seconds
        self.role_poll_seconds = role_poll_seconds
        self._transition_index = 0
        self.journal = JsonlJournal(self.output_dir / "warmup_events.jsonl")
        (self.output_dir / "requests").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "status").mkdir(parents=True, exist_ok=True)

    def _save_json(self, relative_path: str, value: Any) -> None:
        path = self.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _worker_status(self, node: NodeSpec) -> Any:
        return self.client.get_json(node.worker_url, "/pd_flip/runtime_role/status")

    def _router_status(self) -> Any:
        return self.client.get_json(self.router_url, "/pd_flip/router/workers")

    def _next_transition_id(self) -> str:
        self._transition_index += 1
        return "transition-{:03d}".format(self._transition_index)

    def _save_transition_snapshot(
        self,
        action_id: str,
        phase: str,
        kind: str,
        node: NodeSpec,
        response: Any,
    ) -> None:
        relative_path = "status/{}-{}-{}-{}.json".format(
            action_id, kind, node.name, phase
        )
        self._save_json(relative_path, response)
        self.journal.write(
            "transition_snapshot",
            action_id=action_id,
            phase=phase,
            kind=kind,
            node=node.name,
            path=relative_path,
            response=response,
        )

    def _wait_router_worker(
        self,
        node: NodeSpec,
        *,
        expected_role: Optional[str] = None,
        expected_draining: Optional[bool] = None,
    ) -> Any:
        deadline = time.monotonic() + self.role_timeout_seconds
        last_response: Any = None
        while time.monotonic() < deadline:
            last_response = self._router_status()
            workers = {
                str(item.get("worker_id")): item
                for item in _router_workers(last_response)
                if item.get("worker_id") is not None
            }
            item = workers.get(node.router_worker_id)
            if item is not None:
                role_ok = expected_role is None or _router_role(item) == expected_role
                draining_ok = (
                    expected_draining is None
                    or bool(item.get("draining")) is expected_draining
                )
                if role_ok and draining_ok:
                    self.journal.write(
                        "router_worker_ready",
                        node=node.name,
                        expected_role=expected_role,
                        expected_draining=expected_draining,
                        response=last_response,
                    )
                    return last_response
            time.sleep(self.role_poll_seconds)
        raise TimeoutError(
            "router did not publish {} role={} draining={}: {}".format(
                node.name, expected_role, expected_draining, last_response
            )
        )

    def _wait_warmup_target(self, node: NodeSpec) -> Any:
        deadline = time.monotonic() + self.role_timeout_seconds
        last_response: Any = None
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            last_response = self._router_status()
            try:
                validate_router_warmup_target(
                    last_response, target_worker_id=node.router_worker_id
                )
                self.journal.write(
                    "router_warmup_target_ready",
                    node=node.name,
                    response=last_response,
                )
                return last_response
            except ValueError as exc:
                last_error = exc
                time.sleep(self.role_poll_seconds)
        raise TimeoutError(
            "router did not isolate candidate Prefill {}: {}; last={}".format(
                node.name, last_error, last_response
            )
        )

    def _wait_role(self, node: NodeSpec, role: str, *, require_idle: bool = True) -> Any:
        deadline = time.monotonic() + self.role_timeout_seconds
        last_error: Optional[Exception] = None
        last_response: Any = None
        while time.monotonic() < deadline:
            try:
                last_response = self._worker_status(node)
                validate_worker_role(last_response, role, require_idle=require_idle)
                self.journal.write(
                    "role_ready", node=node.name, role=role, response=last_response
                )
                return last_response
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                time.sleep(self.role_poll_seconds)
        raise TimeoutError(
            "worker {} did not reach idle role {}: {}; last={}".format(
                node.name, role, last_error, last_response
            )
        )

    def _set_worker_role(self, node: NodeSpec, role: str) -> None:
        action_id = self._next_transition_id()
        self._save_transition_snapshot(
            action_id, "before", "worker-role", node, self._worker_status(node)
        )
        response = self.client.post_json(
            node.worker_url,
            "/pd_flip/runtime_role/set",
            {"role": role, "force": False},
        )
        self.journal.write(
            "worker_role_set",
            action_id=action_id,
            node=node.name,
            role=role,
            response=response,
        )
        after = self._wait_role(node, role)
        self._save_transition_snapshot(
            action_id, "after", "worker-role", node, after
        )

    def _set_router_role(self, node: NodeSpec, role: str, *, draining: bool) -> None:
        action_id = self._next_transition_id()
        self._save_transition_snapshot(
            action_id, "before", "router-role", node, self._router_status()
        )
        response = self.client.post_json(
            self.router_url,
            "/pd_flip/router/worker/role",
            {
                "worker_id": node.router_worker_id,
                "role": role,
                "bootstrap_port": node.bootstrap_port if role == "prefill" else None,
                "draining": draining,
            },
        )
        self.journal.write(
            "router_role_set",
            action_id=action_id,
            node=node.name,
            role=role,
            draining=draining,
            response=response,
        )
        after = self._wait_router_worker(
            node, expected_role=role, expected_draining=draining
        )
        self._save_transition_snapshot(
            action_id, "after", "router-role", node, after
        )

    def _set_router_draining(self, node: NodeSpec, draining: bool) -> None:
        action_id = self._next_transition_id()
        self._save_transition_snapshot(
            action_id, "before", "router-drain", node, self._router_status()
        )
        response = self.client.post_json(
            self.router_url,
            "/pd_flip/router/worker/drain",
            {"worker_id": node.router_worker_id, "draining": draining},
        )
        self.journal.write(
            "router_drain_set",
            action_id=action_id,
            node=node.name,
            draining=draining,
            response=response,
        )
        after = self._wait_router_worker(node, expected_draining=draining)
        self._save_transition_snapshot(
            action_id, "after", "router-drain", node, after
        )

    def _run_profile(self, node: NodeSpec) -> List[JsonDict]:
        results: List[JsonDict] = []
        for prompt_kind in ("long", "short"):
            trace_row = self.selected_rows[prompt_kind]
            self.journal.write(
                "warmup_request_start",
                node=node.name,
                prompt_kind=prompt_kind,
                trace_request_id=trace_row.get("request_id"),
            )
            result = self.client.stream_chat(self.router_url, trace_row["body"])
            prompt_tokens = int(result["prompt_tokens"])
            if prompt_kind == "long" and prompt_tokens <= 6000:
                raise ValueError("long warmup prompt has only {} tokens".format(prompt_tokens))
            if prompt_kind == "short" and not 500 <= prompt_tokens <= 1000:
                raise ValueError("short warmup prompt has {} tokens".format(prompt_tokens))
            result.update(
                {
                    "node": node.name,
                    "trace_request_id": trace_row.get("request_id"),
                    "trace_prompt_kind": prompt_kind,
                    "trace_prompt_chars": trace_row.get("prompt_chars"),
                    "measured": False,
                    "kv_cache_flushed_after": False,
                }
            )
            self._save_json("requests/{}-{}.json".format(node.name, prompt_kind), result)
            self.journal.write(
                "warmup_request_complete",
                node=node.name,
                prompt_kind=prompt_kind,
                ttft_s=result["ttft_s"],
                total_duration_s=result["total_duration_s"],
                prompt_tokens=prompt_tokens,
            )
            results.append(result)
        return results

    def _warm_initial_prefill(self) -> List[JsonDict]:
        status = self._wait_role(self.initial_prefill, "prefill")
        self._save_json("status/{}-before.json".format(self.initial_prefill.name), status)
        self._wait_warmup_target(self.initial_prefill)
        return self._run_profile(self.initial_prefill)

    def _restore_candidate_to_decode(self, node: NodeSpec) -> None:
        self._set_router_draining(node, True)
        self._set_worker_role(node, "decode")
        self._set_router_role(node, "decode", draining=True)
        self._set_router_draining(node, False)
        self._set_router_draining(self.initial_prefill, False)
        status = self._wait_role(node, "decode")
        self._save_json("status/{}-restored.json".format(node.name), status)

    def _warm_decode_candidate(self, node: NodeSpec) -> List[JsonDict]:
        initial = self._wait_role(node, "decode")
        self._save_json("status/{}-before.json".format(node.name), initial)
        self._set_router_draining(node, True)
        self._set_worker_role(node, "prefill")
        self._set_router_role(node, "prefill", draining=True)
        self._set_router_draining(self.initial_prefill, True)
        self._set_router_draining(node, False)
        self._wait_warmup_target(node)
        try:
            return self._run_profile(node)
        finally:
            self._restore_candidate_to_decode(node)

    def _best_effort_restore(self) -> None:
        self.journal.write("restoration_start")
        failed = False

        # First make every worker unroutable. Role restoration is unsafe until
        # the router can no longer send traffic to any partially restored node.
        for node in self.candidates:
            try:
                self._set_router_draining(node, True)
            except Exception as exc:  # forensic cleanup must continue across nodes
                failed = True
                self.journal.write(
                    "restoration_error", node=node.name, error=repr(exc)
                )

        if failed:
            self.journal.write("restoration_complete", success=False, all_drained=False)
            return

        for node in self.candidates:
            role = "prefill" if node.name == self.initial_prefill.name else "decode"
            try:
                self._set_worker_role(node, role)
            except Exception as exc:
                failed = True
                self.journal.write(
                    "restoration_error", node=node.name, phase="worker_role", error=repr(exc)
                )

        for node in self.candidates:
            role = "prefill" if node.name == self.initial_prefill.name else "decode"
            try:
                self._set_router_role(node, role, draining=True)
            except Exception as exc:
                failed = True
                self.journal.write(
                    "restoration_error", node=node.name, phase="router_role", error=repr(exc)
                )

        if failed:
            self.journal.write("restoration_complete", success=False, all_drained=True)
            return

        for node in self.candidates:
            try:
                self._set_router_draining(node, False)
            except Exception as exc:
                failed = True
                self.journal.write(
                    "restoration_error", node=node.name, phase="undrain", error=repr(exc)
                )
        self.journal.write("restoration_complete", success=not failed)

    def _mark_results_flushed(self, results: Sequence[JsonDict]) -> None:
        for result in results:
            result["kv_cache_flushed_after"] = True
            self._save_json(
                "requests/{}-{}.json".format(
                    result["node"], result["trace_prompt_kind"]
                ),
                result,
            )

    def _flush_and_validate(self) -> None:
        for node in self.nodes.values():
            response = self.client.post_text(node.worker_url, "/flush_cache")
            if "cache flushed" not in response.lower():
                raise ValueError(
                    "worker {} cache flush response was not successful: {}".format(
                        node.name, response
                    )
                )
            self._save_json(
                "status/flush-{}.json".format(node.name),
                {"node": node.name, "success": True, "response": response},
            )
            self.journal.write("cache_flushed", node=node.name)
        for node in self.nodes.values():
            expected_role = "prefill" if node.name == self.initial_prefill.name else "decode"
            response = self._wait_role(node, expected_role)
            self._save_json("status/{}-final.json".format(node.name), response)
            server_info = self.client.get_json(node.worker_url, "/server_info")
            validate_worker_fsm_safe(server_info)
            self._save_json(
                "status/{}-fsm-final.json".format(node.name),
                {"internal_states": server_info.get("internal_states")},
            )
        router = self.client.get_json(self.router_url, "/pd_flip/router/workers")
        validate_router_topology(
            router,
            expected_prefill=getattr(self, "expected_prefill", 1),
            expected_decode=getattr(self, "expected_decode", len(self.nodes) - 1),
        )
        self._save_json("status/router-final.json", router)

    def run(self) -> JsonDict:
        expected_prefill = getattr(self, "expected_prefill", 1)
        expected_decode = getattr(self, "expected_decode", len(self.nodes) - 1)
        expected_topology = getattr(
            self, "expected_topology", "{}P{}D".format(expected_prefill, expected_decode)
        )
        initial_router = self.client.get_json(self.router_url, "/pd_flip/router/workers")
        validate_router_topology(
            initial_router,
            expected_prefill=expected_prefill,
            expected_decode=expected_decode,
        )
        self._save_json("status/router-initial.json", initial_router)
        self.journal.write(
            "candidate_prefill_warmup_start",
            candidates=[node.name for node in self.candidates],
        )
        results: List[JsonDict] = []
        try:
            for node in self.candidates:
                if node.name == self.initial_prefill.name:
                    results.extend(self._warm_initial_prefill())
                else:
                    results.extend(self._warm_decode_candidate(node))
            self._flush_and_validate()
            self._mark_results_flushed(results)
        except Exception as exc:
            self.journal.write("candidate_prefill_warmup_failed", error=repr(exc))
            self._best_effort_restore()
            raise
        summary = {
            "success": True,
            "measured": False,
            "initial_topology": expected_topology,
            "final_topology": expected_topology,
            "candidate_names": [node.name for node in self.candidates],
            "warmup_request_count": len(results),
            "warmup_requests": results,
            "kv_cache_flushed_after": True,
            "completed_utc": utc_now(),
        }
        self._save_json("summary.json", summary)
        self.journal.write("candidate_prefill_warmup_complete", request_count=len(results))
        return summary


def load_trace(path: Path) -> List[JsonDict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--node", action="append", required=True)
    parser.add_argument("--initial-prefill-name", required=True)
    parser.add_argument("--candidate-prefill-name", action="append", required=True)
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="ADMIN_API_KEY")
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--role-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--role-poll-seconds", type=float, default=0.25)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit("{} must be set".format(args.api_key_env))
    nodes = [parse_node_spec(raw) for raw in args.node]
    names = [node.name for node in nodes]
    if len(set(names)) != len(names):
        raise SystemExit("node names must be unique")
    if set(args.candidate_prefill_name) != set(names):
        raise SystemExit("candidate Prefill names must contain every configured node once")
    if len(set(args.candidate_prefill_name)) != len(args.candidate_prefill_name):
        raise SystemExit("candidate Prefill names must be unique")
    selected_rows = select_warmup_rows(load_trace(args.trace_jsonl))
    runner = CandidatePrefillWarmup(
        router_url=args.router_url,
        nodes=nodes,
        initial_prefill_name=args.initial_prefill_name,
        candidate_names=args.candidate_prefill_name,
        selected_rows=selected_rows,
        output_dir=args.output_dir,
        client=HttpClient(api_key, args.request_timeout_seconds),
        role_timeout_seconds=args.role_timeout_seconds,
        role_poll_seconds=args.role_poll_seconds,
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
