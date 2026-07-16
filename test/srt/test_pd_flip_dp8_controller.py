import pytest

from scripts.playground.disaggregation.pd_flip_controller import (
    _assign_target_dp_ranks,
    _index_dp_responses,
    _migration_response_complete,
    _migration_response_failed,
    _migration_source_start_payload,
    _order_manifests_by_requested_rids,
    _require_request_owners,
    _require_worker_dp_ranks,
    _request_owner_map,
    select_target_dp_rank,
)


def test_source_manifests_are_restored_to_controller_request_order():
    manifests = [
        {"rid": "rank-3-rid", "source_decode_dp_rank": 3},
        {"rid": "waiting-rid", "source_decode_dp_rank": 7},
        {"rid": "rank-2-rid", "source_decode_dp_rank": 2},
    ]

    ordered = _order_manifests_by_requested_rids(
        manifests, ["rank-2-rid", "rank-3-rid"]
    )

    assert [manifest["rid"] for manifest in ordered] == [
        "rank-2-rid",
        "rank-3-rid",
        "waiting-rid",
    ]


def test_index_dp_responses_requires_unique_ranks():
    with pytest.raises(RuntimeError, match="duplicate dp_rank"):
        _index_dp_responses([{"dp_rank": 1}, {"dp_rank": 1}])


def test_index_dp_responses_accepts_nested_status_and_dp1_fallback():
    assert _index_dp_responses({"status": {"role": "decode"}}) == {
        0: {"status": {"role": "decode"}}
    }
    assert sorted(
        _index_dp_responses(
            [
                {"status": {"dp_rank": 3}},
                {"dp_rank": 5, "status": {"dp_rank": 5}},
            ]
        )
    ) == [3, 5]


def test_index_dp_responses_rejects_missing_rank_in_multi_response():
    with pytest.raises(RuntimeError, match="missing dp_rank"):
        _index_dp_responses([{"dp_rank": 0}, {"status": {"role": "decode"}}])


def test_request_owner_map_requires_exactly_one_owner():
    responses = [
        {"dp_rank": 0, "handled_rids": ["r1"]},
        {"dp_rank": 1, "handled_rids": ["r1"]},
    ]
    with pytest.raises(RuntimeError, match="multiple owners"):
        _request_owner_map(responses, "handled_rids")


def test_request_owner_map_indexes_each_handled_request():
    responses = [
        {"dp_rank": 0, "handled_rids": ["r0", "r2"]},
        {"status": {"dp_rank": 1}, "handled_rids": ["r1"]},
    ]
    assert _request_owner_map(responses, "handled_rids") == {
        "r0": 0,
        "r1": 1,
        "r2": 0,
    }


def test_target_selection_uses_free_kv_capacity():
    statuses = [
        {"dp_rank": 0, "free_kv_pages": 2},
        {"dp_rank": 1, "free_kv_pages": 20},
    ]
    assert select_target_dp_rank(statuses, required_pages=8) == 1


def test_target_selection_filters_role_admission_and_request_capacity():
    statuses = [
        {
            "dp_rank": 0,
            "role": "prefill",
            "free_request_slots": 8,
            "free_kv_pages": 100,
        },
        {
            "dp_rank": 1,
            "role": "decode",
            "admission_paused": True,
            "free_request_slots": 8,
            "free_kv_pages": 90,
        },
        {
            "dp_rank": 2,
            "role": "decode",
            "free_request_slots": 0,
            "free_kv_pages": 80,
        },
        {
            "dp_rank": 3,
            "role": "decode",
            "free_request_slots": 1,
            "free_kv_pages": 10,
        },
    ]
    assert select_target_dp_rank(statuses, required_pages=8) == 3


def test_target_selection_breaks_capacity_ties_by_lowest_rank():
    statuses = [
        {"dp_rank": 7, "free_request_slots": 1, "free_kv_pages": 20},
        {"dp_rank": 2, "free_request_slots": 1, "free_kv_pages": 20},
    ]
    assert select_target_dp_rank(statuses, required_pages=8) == 2


def test_target_selection_rejects_insufficient_capacity():
    with pytest.raises(RuntimeError, match="no decode DP rank"):
        select_target_dp_rank(
            [{"dp_rank": 0, "free_request_slots": 1, "free_kv_pages": 7}],
            required_pages=8,
        )


def test_source_start_payload_carries_target_rank_before_sender_creation():
    payload = _migration_source_start_payload(
        "session",
        "http://target",
        ["r1"],
        target_decode_dp_rank=6,
    )
    assert payload["target_decode_dp_rank"] == 6


def test_source_start_payload_carries_per_request_target_ranks():
    payload = _migration_source_start_payload(
        "session",
        "http://target",
        ["r0", "r1"],
        target_decode_dp_ranks={"r0": 2, "r1": 6},
    )
    assert payload["target_decode_dp_ranks"] == {"r0": 2, "r1": 6}


def test_target_assignment_consumes_rank_pages_and_request_slots():
    statuses = [
        {"dp_rank": 0, "free_request_slots": 1, "free_kv_pages": 12},
        {"dp_rank": 1, "free_request_slots": 2, "free_kv_pages": 10},
    ]
    assert _assign_target_dp_ranks(
        statuses, {"r0": 8, "r1": 8, "r2": 2}
    ) == {"r0": 0, "r1": 1, "r2": 1}


def test_migration_barrier_waits_for_every_dp_rank():
    responses = [
        {"dp_rank": 0, "status": {"pending_reqs": 0, "failed_reqs": 0}},
        {"dp_rank": 1, "status": {"pending_reqs": 1, "failed_reqs": 0}},
    ]
    assert not _migration_response_complete(responses)


def test_migration_barrier_detects_failure_on_nonfirst_dp_rank():
    responses = [
        {"dp_rank": 0, "status": {"pending_reqs": 0, "failed_reqs": 0}},
        {"dp_rank": 1, "status": {"pending_reqs": 0, "failed_reqs": 1}},
    ]
    assert _migration_response_failed(responses)


def test_request_owner_barrier_reports_missing_rids():
    responses = [
        {"dp_rank": 0, "handled_rids": ["r0"]},
        {"dp_rank": 1, "handled_rids": [], "ignored_rids": ["r0", "r1"]},
    ]
    with pytest.raises(RuntimeError, match=r"missing_rids=\['r1'\]"):
        _require_request_owners(responses, ["r0", "r1"], "target commit")


def test_worker_barrier_reports_a_missing_dp_rank():
    expected = [{"dp_rank": 0}, {"dp_rank": 1}]
    with pytest.raises(RuntimeError, match=r"missing_dp_ranks=\[1\]"):
        _require_worker_dp_ranks([{"dp_rank": 0}], expected, "target status")
