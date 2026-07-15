import ast
import pathlib
import textwrap
import types
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEDULER_PATH = REPO_ROOT / "python/sglang/srt/managers/scheduler.py"
SERVER_ARGS_PATH = REPO_ROOT / "python/sglang/srt/server_args.py"
IO_STRUCT_PATH = REPO_ROOT / "python/sglang/srt/managers/io_struct.py"


def _load_class_method(path, class_name, method_name, extra_namespace=None):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        (
            node
            for node in class_def.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        ),
        None,
    )
    if method is None:
        raise AttributeError(f"{class_name} has no method {method_name}")
    namespace = {"Any": Any, "Dict": Dict, "List": List, "Tuple": Tuple}
    namespace.update(extra_namespace or {})
    exec(textwrap.dedent(ast.get_source_segment(source, method)), namespace)
    return namespace[method_name]


@pytest.mark.parametrize(
    "prompt_len,page_size,expected",
    [(0, 64, 0), (63, 64, 0), (64, 64, 64), (1974, 64, 1920)],
)
def test_prefill_donor_boundary_uses_complete_prompt_pages(
    prompt_len, page_size, expected
):
    boundary = _load_class_method(
        SCHEDULER_PATH, "Scheduler", "_pd_flip_prefill_donor_boundary"
    )

    assert boundary(prompt_len, page_size) == expected


def test_prefill_donor_protocol_fields_are_opt_in():
    server_args_source = SERVER_ARGS_PATH.read_text(encoding="utf-8")
    io_struct_source = IO_STRUCT_PATH.read_text(encoding="utf-8")

    assert "enable_pd_flip_prefill_donor: bool = False" in server_args_source
    assert '"--enable-pd-flip-prefill-donor"' in server_args_source
    assert "prefill_donor_mode: bool = False" in io_struct_source


def test_source_manifest_preserves_original_prefill_identity():
    apply_manifest = _load_class_method(
        SCHEDULER_PATH,
        "Scheduler",
        "_pd_flip_apply_prefill_donor_manifest",
        {"Req": object},
    )
    scheduler = types.SimpleNamespace(
        server_args=types.SimpleNamespace(dp_size=1),
        token_to_kv_pool_allocator=types.SimpleNamespace(page_size=64),
        _pd_flip_migration_room_for_req=lambda _req: 700,
    )
    scheduler._pd_flip_prefill_donor_boundary = _load_class_method(
        SCHEDULER_PATH, "Scheduler", "_pd_flip_prefill_donor_boundary"
    )
    scheduler._pd_flip_prefill_donor_room_for_req = types.MethodType(
        _load_class_method(
            SCHEDULER_PATH,
            "Scheduler",
            "_pd_flip_prefill_donor_room_for_req",
            {"Req": object},
        ),
        scheduler,
    )
    req = types.SimpleNamespace(
        origin_input_ids=list(range(1974)),
        bootstrap_host="192.168.0.42",
        bootstrap_port=8998,
    )
    manifest = {"migration_bootstrap_room": 700}

    apply_manifest(scheduler, req, manifest)

    assert manifest["prefill_donor_host"] == "192.168.0.42"
    assert manifest["prefill_donor_port"] == 8998
    assert manifest["prompt_len"] == 1974
    assert manifest["prefill_donor_end"] == 1920
    assert manifest["source_decode_start"] == 1920
    assert manifest["prefill_donor_bootstrap_room"] != 700


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, item):
        value = self.values[item]
        return _Tensor(value) if isinstance(value, np.ndarray) else value

    def __len__(self):
        return len(self.values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Receiver:
    def __init__(self):
        self.metadata_calls = []

    def send_metadata(
        self, page_indices, metadata_index, state_indices, decode_prefix_len
    ):
        self.metadata_calls.append(
            (
                list(page_indices),
                metadata_index,
                state_indices,
                decode_prefix_len,
            )
        )


class _Allocator:
    def __init__(self, values):
        self.values = iter(values)

    def alloc(self):
        return next(self.values)


class _FreeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, value):
        self.freed.append(value)


def _page_indices(kv_indices, page_size):
    values = np.asarray(kv_indices)
    return values[::page_size] // page_size


def test_target_donor_mode_skips_target_prefix_match_and_splits_ranges():
    prealloc = _load_class_method(
        SCHEDULER_PATH,
        "Scheduler",
        "_pd_flip_target_prealloc_donor_ranges",
        {
            "Any": Any,
            "DecodeRequest": object,
            "Dict": Dict,
            "kv_to_page_indices": _page_indices,
        },
    )
    source_receiver = _Receiver()
    prefill_receiver = _Receiver()
    req = types.SimpleNamespace(
        rid="r0",
        req_pool_idx=None,
        kv_committed_len=2993,
        origin_input_ids=list(range(1974)),
        cache_protected_len=None,
    )
    prefill_req = types.SimpleNamespace(bootstrap_room=900)
    prefix_calls = []

    def _pre_alloc(req, **kwargs):
        req.req_pool_idx = 0
        assert kwargs == {
            "prefix_len": 0,
            "total_prefix_len": 0,
            "fill_len_override": 2993,
        }
        return _Tensor(np.arange(2993))

    queue = types.SimpleNamespace(
        _pre_alloc=_pre_alloc,
        _match_prefix_and_lock=lambda _req: prefix_calls.append(_req),
        kv_manager=types.SimpleNamespace(
            kv_args=types.SimpleNamespace(state_types=[])
        ),
    )
    scheduler = types.SimpleNamespace(
        disagg_decode_prealloc_queue=queue,
        req_to_metadata_buffer_idx_allocator=_Allocator([7, 8]),
        req_to_token_pool=types.SimpleNamespace(
            req_to_token=_Tensor([np.arange(2993)])
        ),
        token_to_kv_pool_allocator=types.SimpleNamespace(page_size=64),
        _pd_flip_target_state_indices=lambda _req, _end: [],
    )
    entry = {
        "decode_req": types.SimpleNamespace(req=req, kv_receiver=source_receiver),
        "prefill_decode_req": types.SimpleNamespace(
            req=prefill_req, kv_receiver=prefill_receiver
        ),
        "manifest": {
            "prompt_len": 1974,
            "prefill_donor_end": 1920,
            "source_decode_start": 1920,
            "kv_committed_len": 2993,
        },
        "metadata_index": -1,
        "prefill_metadata_index": -1,
    }

    prealloc(scheduler, entry)

    assert prefix_calls == []
    assert entry["target_prefix_match_skipped"] is True
    assert entry["prefill_received_start"] == 0
    assert entry["prefill_received_end"] == 1920
    assert entry["source_transfer_start"] == 1920
    assert entry["source_transfer_end"] == 2993
    assert prefill_receiver.metadata_calls[0][1:] == (8, [], 0)
    assert source_receiver.metadata_calls[0][1:] == (7, [], 1920)


def test_target_donor_coverage_rejects_uninitialized_slot():
    ready = _load_class_method(
        SCHEDULER_PATH,
        "Scheduler",
        "_pd_flip_target_donor_ranges_ready",
        {"Any": Any, "Dict": Dict},
    )
    req = types.SimpleNamespace(rid="r0", req_pool_idx=0)
    mapping = np.arange(20)
    mapping[9] = 0
    scheduler = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(req_to_token=_Tensor([mapping])),
        _pd_flip_invalid_kv_positions=lambda values: [
            i for i, value in enumerate(values.values) if value <= 0
        ],
    )
    entry = {
        "decode_req": types.SimpleNamespace(req=req),
        "target_prompt_len": 10,
        "target_committed_len": 20,
        "prefill_received_start": 0,
        "prefill_received_end": 8,
        "source_transfer_start": 8,
        "source_transfer_end": 20,
    }

    with pytest.raises(RuntimeError, match="uninitialized KV indices"):
        ready(scheduler, entry)


def test_target_transfer_dispatches_to_prefill_donor_pump():
    pump = _load_class_method(
        SCHEDULER_PATH,
        "Scheduler",
        "_pd_flip_target_pump_transfer",
        {"Any": Any, "Dict": Dict},
    )
    calls = []
    scheduler = types.SimpleNamespace(
        _pd_flip_target_pump_donor_transfer=lambda session: calls.append(session)
    )
    session = {"prefill_donor_mode": True, "target_entries": {}}

    pump(scheduler, session)

    assert calls == [session]


class _Poll:
    Failed = "failed"
    WaitingForInput = "waiting"
    Transferring = "transferring"
    Success = "success"


class _PollingReceiver:
    def __init__(self, polls):
        self.polls = iter(polls)
        self.cleared = False
        self.aborted = False

    def poll(self):
        return next(self.polls)

    def clear(self):
        self.cleared = True

    def abort(self):
        self.aborted = True


def test_target_donor_pump_holds_request_after_both_ranges_arrive():
    pump = _load_class_method(
        SCHEDULER_PATH,
        "Scheduler",
        "_pd_flip_target_pump_donor_transfer",
        {"Any": Any, "Dict": Dict, "KVPoll": _Poll, "time": types.SimpleNamespace(monotonic=lambda: 1.0)},
    )
    source_receiver = _PollingReceiver([_Poll.WaitingForInput, _Poll.Success])
    prefill_receiver = _PollingReceiver([_Poll.WaitingForInput, _Poll.Success])
    entry = {
        "decode_req": types.SimpleNamespace(
            req=types.SimpleNamespace(rid="r0"), kv_receiver=source_receiver
        ),
        "prefill_decode_req": types.SimpleNamespace(
            req=types.SimpleNamespace(rid="r0"), kv_receiver=prefill_receiver
        ),
        "phase": "new",
    }
    notes = []
    scheduler = types.SimpleNamespace(
        _pd_flip_target_init_receiver=lambda _decode_req: True,
        _pd_flip_target_prealloc_donor_ranges=lambda _entry: None,
        _pd_flip_target_donor_ranges_ready=lambda _entry: True,
        _pd_flip_target_metadata_ready_for=lambda _entry, _key, _decode_req: True,
        _pd_flip_free_target_metadata=lambda _entry: None,
        _pd_flip_release_target_request=lambda _entry: None,
        _pd_flip_note_timing=lambda _entry, name, *args: notes.append(name),
        _pd_flip_target_pump_delta_transfer=lambda _session: None,
    )
    session = {
        "target_entries": {"r0": entry},
        "manifests": [{"rid": "r0"}],
        "prepare_only": True,
        "state": "target_prepared",
    }

    pump(scheduler, session)

    assert entry["phase"] == "transferred_held"
    assert entry["held"] is True
    assert session["state"] == "target_transferred_held"
    assert session["transferred_rids"] == {"r0"}
    assert source_receiver.cleared is True
    assert prefill_receiver.cleared is True


def test_target_metadata_ready_for_checks_the_selected_receiver_room():
    ready_for = _load_class_method(
        SCHEDULER_PATH,
        "Scheduler",
        "_pd_flip_target_metadata_ready_for",
        {"Any": Any, "Dict": Dict},
    )
    rooms = np.zeros((10, 1), dtype=np.int64)
    scheduler = types.SimpleNamespace(
        disagg_metadata_buffers=types.SimpleNamespace(bootstrap_room=rooms)
    )
    receiver_req = types.SimpleNamespace(
        req=types.SimpleNamespace(bootstrap_room=900)
    )
    entry = {"prefill_metadata_index": 8}

    assert ready_for(scheduler, entry, "prefill_metadata_index", receiver_req) is False
    rooms[8, 0] = 900
    assert ready_for(scheduler, entry, "prefill_metadata_index", receiver_req) is True
    rooms[8, 0] = 901
    with pytest.raises(RuntimeError, match="expected 900, got 901"):
        ready_for(scheduler, entry, "prefill_metadata_index", receiver_req)


def test_target_metadata_cleanup_frees_source_and_prefill_buffers_once():
    cleanup = _load_class_method(
        SCHEDULER_PATH,
        "Scheduler",
        "_pd_flip_free_target_metadata",
        {"Any": Any, "Dict": Dict},
    )
    rooms = np.ones((10, 1), dtype=np.int64)
    allocator = _FreeAllocator()
    scheduler = types.SimpleNamespace(
        disagg_metadata_buffers=types.SimpleNamespace(bootstrap_room=rooms),
        req_to_metadata_buffer_idx_allocator=allocator,
    )
    entry = {"metadata_index": 7, "prefill_metadata_index": 8}

    cleanup(scheduler, entry)
    cleanup(scheduler, entry)

    assert allocator.freed == [7, 8]
    assert entry == {"metadata_index": -1, "prefill_metadata_index": -1}
    assert rooms[7, 0] == 0
    assert rooms[8, 0] == 0


def test_target_donor_entry_uses_source_d_and_original_p_bootstrap():
    prepare = _load_class_method(
        SCHEDULER_PATH,
        "Scheduler",
        "_pd_flip_prepare_target_donor_entry",
        {
            "Any": Any,
            "Dict": Dict,
            "DecodeRequest": lambda req, kv_receiver: types.SimpleNamespace(
                req=req, kv_receiver=kv_receiver
            ),
        },
    )

    class Receiver:
        def __init__(self, mgr, bootstrap_addr, bootstrap_room):
            self.mgr = mgr
            self.bootstrap_addr = bootstrap_addr
            self.bootstrap_room = bootstrap_room

    def manifest_to_req(manifest, host):
        return types.SimpleNamespace(
            rid=manifest["rid"],
            bootstrap_host=host,
            bootstrap_port=manifest["source_bootstrap_port"],
            bootstrap_room=manifest["migration_bootstrap_room"],
        )

    scheduler = types.SimpleNamespace(_pd_flip_manifest_to_req=manifest_to_req)
    manifest = {
        "rid": "r0",
        "source_bootstrap_port": 8998,
        "migration_bootstrap_room": 700,
        "prefill_donor_host": "prefill-p",
        "prefill_donor_port": 8999,
        "prefill_donor_bootstrap_room": 1700,
        "prefill_donor_end": 1920,
        "kv_committed_len": 2993,
        "pd_flip_source_queue": "running",
    }
    kv_manager = object()

    entry = prepare(scheduler, manifest, "source-d", Receiver, kv_manager)

    assert entry["decode_req"].kv_receiver.bootstrap_addr == "source-d:8998"
    assert entry["decode_req"].kv_receiver.bootstrap_room == 700
    assert entry["prefill_decode_req"].kv_receiver.bootstrap_addr == "prefill-p:8999"
    assert entry["prefill_decode_req"].kv_receiver.bootstrap_room == 1700
    assert entry["prefill_decode_req"].req.bootstrap_room == 1700
    assert entry["prefill_metadata_index"] == -1
