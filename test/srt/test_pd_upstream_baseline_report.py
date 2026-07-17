import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_upstream_baseline_report.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("pd_upstream_baseline_report", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_run(root: Path) -> Path:
    manifest = {
        "run_id": "unit-run",
        "mode": "upstream_baseline",
        "validity": "pending",
        "image": "tiancij/sglang-upstream:v0.5.15-clean",
        "image_id": "sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e",
        "trace_sha256": "82da848d68c9662a7aaaf76deb547b1d8cc6c4f562586f0d60dd212bc114e964",
        "model_id": "Qwen3-Next-80B-A3B-Instruct",
        "model_fingerprint": "model-sha",
        "router_sha256": "router-sha",
        "topology": "1P3D",
        "gpu_ids": "0,1,2,3",
        "tp_size": 4,
        "dp_size": 1,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        {
            "request_id": "qwen80b-00",
            "prompt_kind": "long",
            "arrival_offset_s": 0.0,
            "status": "completed",
            "completion_tokens": 3,
            "completion_token_match": True,
            "finish_reason": "length",
            "ttft_s": 4.0,
            "ttft_slo_s": 5.0,
            "ttft_met": True,
            "avg_tpot_s": 0.03,
            "p50_tpot_s": 0.03,
            "p95_tpot_s": 0.04,
            "max_tpot_s": 0.04,
            "good_tpot_intervals": 2,
            "total_tpot_intervals": 2,
            "latency_s": 4.06,
        },
        {
            "request_id": "qwen80b-01",
            "prompt_kind": "short",
            "arrival_offset_s": 0.5,
            "status": "completed",
            "completion_tokens": 3,
            "completion_token_match": True,
            "finish_reason": "length",
            "ttft_s": 3.0,
            "ttft_slo_s": 2.0,
            "ttft_met": False,
            "avg_tpot_s": 0.055,
            "p50_tpot_s": 0.04,
            "p95_tpot_s": 0.07,
            "max_tpot_s": 0.07,
            "good_tpot_intervals": 1,
            "total_tpot_intervals": 2,
            "latency_s": 3.11,
        },
    ]
    raw = root / "raw" / "upstream_baseline"
    write_jsonl(raw / "request_metrics.jsonl", rows)
    write_jsonl(raw / "errors.jsonl", [])
    write_jsonl(root / "raw" / "slo_ledger.jsonl", [{"x": i} for i in range(8)])
    with (raw / "tpot_tokens.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["request_id", "interval_index", "interval_s", "tpot_slo_s", "met"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"request_id": "qwen80b-00", "interval_index": 1, "interval_s": 0.02, "tpot_slo_s": 0.05, "met": True},
                {"request_id": "qwen80b-00", "interval_index": 2, "interval_s": 0.04, "tpot_slo_s": 0.05, "met": True},
                {"request_id": "qwen80b-01", "interval_index": 1, "interval_s": 0.04, "tpot_slo_s": 0.05, "met": True},
                {"request_id": "qwen80b-01", "interval_index": 2, "interval_s": 0.07, "tpot_slo_s": 0.05, "met": False},
            ]
        )
    return root


def test_generates_valid_summary_and_client_observed_report(tmp_path):
    module = load_module()
    run = make_run(tmp_path)

    summary = module.generate_report(
        run, expected_requests=2, expected_tokens=3, expected_ledger_rows=8, expected_tpot_rows=4
    )

    assert summary["valid"] is True
    assert summary["ttft"]["all"]["mean_s"] == pytest.approx(3.5)
    assert summary["ttft"]["all"]["attainment"] == pytest.approx(0.5)
    assert summary["ttft"]["long"]["attainment"] == pytest.approx(1.0)
    assert summary["tpot_intervals"]["p50_s"] == pytest.approx(0.04)
    assert summary["tpot_intervals"]["p95_s"] == pytest.approx(0.07)
    assert summary["tpot_intervals"]["attainment"] == pytest.approx(0.75)
    report = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert "client-observed" in report
    assert "one measured run" in report
    assert "qwen80b-00" in (run / "report" / "ttft_scatter.svg").read_text(encoding="utf-8")
    assert "qwen80b-01" in (run / "report" / "tpot_scatter.svg").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[1].update(request_id=rows[0]["request_id"]), "duplicate request IDs"),
        (lambda rows: rows[0].update(status="failed"), "not completed"),
        (lambda rows: rows[0].update(completion_tokens=2), "completion tokens"),
        (lambda rows: rows[0].update(completion_token_match=False), "forced output"),
        (lambda rows: rows[0].update(finish_reason="stop"), "finish reason"),
    ],
)
def test_rejects_request_integrity_failures(tmp_path, mutation, message):
    module = load_module()
    run = make_run(tmp_path)
    path = run / "raw" / "upstream_baseline" / "request_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutation(rows)
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match=message):
        module.generate_report(
            run, expected_requests=2, expected_tokens=3, expected_ledger_rows=8, expected_tpot_rows=4
        )


def test_rejects_nonempty_errors_and_raw_row_count_mismatch(tmp_path):
    module = load_module()
    run = make_run(tmp_path)
    write_jsonl(run / "raw" / "upstream_baseline" / "errors.jsonl", [{"error": "boom"}])

    with pytest.raises(ValueError, match="errors.jsonl"):
        module.generate_report(
            run, expected_requests=2, expected_tokens=3, expected_ledger_rows=8, expected_tpot_rows=4
        )

    write_jsonl(run / "raw" / "upstream_baseline" / "errors.jsonl", [])
    with pytest.raises(ValueError, match="ledger row count"):
        module.generate_report(
            run, expected_requests=2, expected_tokens=3, expected_ledger_rows=9, expected_tpot_rows=4
        )


def test_rejects_missing_provenance(tmp_path):
    module = load_module()
    run = make_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    del manifest["router_sha256"]
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="router_sha256"):
        module.generate_report(
            run, expected_requests=2, expected_tokens=3, expected_ledger_rows=8, expected_tpot_rows=4
        )
