import types

import pytest

from sglang.srt.managers.scheduler import Scheduler


def make_scheduler(*, attn_dp_rank, dp_size):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.ps = types.SimpleNamespace(attn_dp_rank=attn_dp_rank, dp_rank=0)
    scheduler.server_args = types.SimpleNamespace(dp_size=dp_size)
    return scheduler


def test_rank_filter_selects_only_local_owner():
    scheduler = make_scheduler(attn_dp_rank=3, dp_size=8)
    manifests = [
        {"rid": "a", "source_decode_dp_rank": 3},
        {"rid": "b", "source_decode_dp_rank": 6},
    ]

    selected = scheduler._pd_flip_manifests_for_rank(manifests, "source_decode_dp_rank")

    assert [manifest["rid"] for manifest in selected] == ["a"]


def test_rank_filter_rejects_missing_rank_in_dp8():
    scheduler = make_scheduler(attn_dp_rank=3, dp_size=8)

    with pytest.raises(ValueError, match="source_decode_dp_rank"):
        scheduler._pd_flip_manifests_for_rank([{"rid": "a"}], "source_decode_dp_rank")


def test_rank_filter_keeps_legacy_dp1_manifest():
    scheduler = make_scheduler(attn_dp_rank=0, dp_size=1)

    selected = scheduler._pd_flip_manifests_for_rank(
        [{"rid": "legacy"}], "target_decode_dp_rank"
    )

    assert [manifest["rid"] for manifest in selected] == ["legacy"]


def test_rank_partition_reports_handled_and_ignored_rids():
    scheduler = make_scheduler(attn_dp_rank=1, dp_size=8)
    manifests = [
        {"rid": "local", "target_decode_dp_rank": 1},
        {"rid": "remote", "target_decode_dp_rank": 4},
    ]

    partition = scheduler._pd_flip_partition_manifests(
        manifests, "target_decode_dp_rank"
    )

    assert partition["dp_rank"] == 1
    assert [manifest["rid"] for manifest in partition["manifests"]] == ["local"]
    assert partition["handled_rids"] == ["local"]
    assert partition["ignored_rids"] == ["remote"]


def test_rid_partition_ignores_requests_owned_by_other_ranks():
    scheduler = make_scheduler(attn_dp_rank=1, dp_size=8)

    partition = scheduler._pd_flip_partition_rids(["local", "remote"], ["local"])

    assert partition["handled_rids"] == ["local"]
    assert partition["ignored_rids"] == ["remote"]


def test_transfer_destination_uses_manifest_target_rank():
    scheduler = make_scheduler(attn_dp_rank=1, dp_size=8)
    scheduler.ps.tp_rank = 3

    assert scheduler._pd_flip_dest_ranks({"target_decode_dp_rank": 6}) == [6]


def test_transfer_destination_requires_target_rank_in_dp8():
    scheduler = make_scheduler(attn_dp_rank=1, dp_size=8)
    scheduler.ps.tp_rank = 3

    with pytest.raises(ValueError, match="target_decode_dp_rank"):
        scheduler._pd_flip_dest_ranks({})


def test_transfer_destination_keeps_local_tp_rank_for_dp1():
    scheduler = make_scheduler(attn_dp_rank=0, dp_size=1)
    scheduler.ps.tp_rank = 3

    assert scheduler._pd_flip_dest_ranks({}) == [3]
