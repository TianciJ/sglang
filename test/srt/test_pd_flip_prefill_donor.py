import ast
import pathlib
import textwrap
import types
from typing import Any, Dict, List, Tuple

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
