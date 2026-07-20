import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_candidate_prefill_warmup.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "pd_flip_candidate_prefill_warmup_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CandidatePrefillWarmupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def require_api(self, name):
        self.assertTrue(hasattr(self.module, name), f"missing warmup API: {name}")
        return getattr(self.module, name)

    def test_select_warmup_rows_copies_long_and_short_without_custom_processor(self):
        rows = [
            {
                "request_id": "short-1",
                "prompt_kind": "short",
                "body": {
                    "messages": [{"role": "user", "content": "short"}],
                    "max_tokens": 10000,
                    "custom_params": {"forced_text": "x"},
                    "custom_logit_processor": "serialized-processor",
                },
            },
            {
                "request_id": "long-1",
                "prompt_kind": "long",
                "body": {
                    "messages": [{"role": "user", "content": "long"}],
                    "max_tokens": 10000,
                    "custom_params": {"forced_text": "x"},
                    "custom_logit_processor": "serialized-processor",
                },
            },
        ]

        selected = self.require_api("select_warmup_rows")(rows)

        self.assertEqual(tuple(selected), ("long", "short"))
        for kind in ("long", "short"):
            self.assertEqual(selected[kind]["body"]["max_tokens"], 1)
            self.assertTrue(selected[kind]["body"]["stream"])
            self.assertEqual(
                selected[kind]["body"]["stream_options"], {"include_usage": True}
            )
            self.assertNotIn("custom_params", selected[kind]["body"])
            self.assertNotIn("custom_logit_processor", selected[kind]["body"])
        self.assertEqual(rows[0]["body"]["max_tokens"], 10000)

    def test_select_warmup_rows_rejects_an_incomplete_profile(self):
        with self.assertRaisesRegex(ValueError, "missing warmup trace kind: short"):
            self.require_api("select_warmup_rows")(
                [
                    {
                        "request_id": "long-1",
                        "prompt_kind": "long",
                        "body": {"messages": []},
                    }
                ]
            )

    def test_parse_node_spec_requires_unique_complete_fields(self):
        parse_node_spec = self.require_api("parse_node_spec")
        node = parse_node_spec(
            "name=node2,worker_url=http://192.0.2.2:30000,"
            "router_worker_id=http://192.0.2.2:30000,bootstrap_port=8998"
        )
        self.assertEqual(node.name, "node2")
        self.assertEqual(node.bootstrap_port, 8998)

        with self.assertRaisesRegex(ValueError, "missing node fields"):
            parse_node_spec("name=node2,worker_url=http://worker")

    def test_validate_worker_role_requires_every_shard_and_active_loop(self):
        good = [
            {
                "success": True,
                "status": {
                    "role": "prefill",
                    "active_event_loop_role": "prefill",
                    "is_idle": True,
                    "running_requests": [],
                    "waiting_requests": [],
                },
            },
            {
                "success": True,
                "status": {
                    "role": "prefill",
                    "active_event_loop_role": "prefill",
                    "is_idle": True,
                    "running_requests": [],
                    "waiting_requests": [],
                },
            },
        ]
        validate_worker_role = self.require_api("validate_worker_role")
        validate_worker_role(good, "prefill", require_idle=True)

        bad = json.loads(json.dumps(good))
        bad[1]["status"]["active_event_loop_role"] = "decode"
        with self.assertRaisesRegex(ValueError, "active event loop"):
            validate_worker_role(bad, "prefill", require_idle=True)

    def test_validate_router_topology_requires_restored_1p3d_without_draining(self):
        workers = {
            "workers": [
                {"worker_id": "http://n0:30000", "role": "prefill", "draining": False},
                {"worker_id": "http://n1:30000", "role": "decode", "draining": False},
                {"worker_id": "http://n2:30000", "role": "decode", "draining": False},
                {"worker_id": "http://n3:30000", "role": "decode", "draining": False},
            ]
        }
        validate_router_topology = self.require_api("validate_router_topology")
        validate_router_topology(workers, expected_prefill=1, expected_decode=3)

        workers["workers"][2]["draining"] = True
        with self.assertRaisesRegex(ValueError, "draining"):
            validate_router_topology(
                workers, expected_prefill=1, expected_decode=3
            )

    def test_validate_router_warmup_target_requires_one_routable_prefill(self):
        workers = {
            "workers": [
                {"worker_id": "http://n0:30000", "role": "prefill", "draining": True},
                {"worker_id": "http://n1:30000", "role": "prefill", "draining": False},
                {"worker_id": "http://n2:30000", "role": "decode", "draining": False},
                {"worker_id": "http://n3:30000", "role": "decode", "draining": False},
            ]
        }
        validate_target = self.require_api("validate_router_warmup_target")
        validate_target(workers, target_worker_id="http://n1:30000")

        workers["workers"][0]["draining"] = False
        with self.assertRaisesRegex(ValueError, "routable Prefill"):
            validate_target(workers, target_worker_id="http://n1:30000")

        workers["workers"][0]["draining"] = True
        workers["workers"][2]["draining"] = True
        with self.assertRaisesRegex(ValueError, "routable Decode"):
            validate_target(workers, target_worker_id="http://n1:30000")

    def test_validate_worker_fsm_safe_rejects_warmup_contamination(self):
        validate_fsm = self.require_api("validate_worker_fsm_safe")
        response = {
            "internal_states": [
                {"pd_flip": {"enabled": True, "state": "safe", "direction": "none"}},
                {"pd_flip": {"enabled": True, "state": "safe", "direction": "none"}},
            ]
        }
        validate_fsm(response)

        response["internal_states"][1]["pd_flip"]["state"] = "preparing"
        with self.assertRaisesRegex(ValueError, "not safe"):
            validate_fsm(response)

    def test_jsonl_journal_appends_timestamped_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            journal = self.require_api("JsonlJournal")(path)
            journal.write("role_ready", node="node2", role="prefill")
            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(row["event"], "role_ready")
        self.assertEqual(row["node"], "node2")
        self.assertEqual(row["role"], "prefill")
        self.assertTrue(row["timestamp_utc"].endswith("+00:00"))

    def test_transition_waits_record_full_status_responses(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for event in ("router_worker_ready", "router_warmup_target_ready", "role_ready"):
            self.assertIn(f'"{event}"', source)
        self.assertGreaterEqual(source.count("response=last_response"), 3)

    def test_best_effort_restore_drains_every_worker_before_role_changes(self):
        warmup = self.require_api("CandidatePrefillWarmup").__new__(
            self.require_api("CandidatePrefillWarmup")
        )
        NodeSpec = self.require_api("NodeSpec")
        warmup.candidates = [
            NodeSpec(f"node{i}", f"http://n{i}", f"worker-{i}", 8998)
            for i in range(4)
        ]
        warmup.initial_prefill = warmup.candidates[0]
        actions = []

        class Journal:
            def write(self, event, **fields):
                actions.append(("journal", event, fields))

        warmup.journal = Journal()
        warmup._set_router_draining = lambda node, draining: actions.append(
            ("drain", node.name, draining)
        )
        warmup._set_worker_role = lambda node, role: actions.append(
            ("worker", node.name, role)
        )
        warmup._set_router_role = lambda node, role, draining: actions.append(
            ("router", node.name, role, draining)
        )
        warmup._best_effort_restore()

        drains = [item for item in actions if item[0] == "drain"]
        workers = [item for item in actions if item[0] == "worker"]
        router_roles = [item for item in actions if item[0] == "router"]
        self.assertEqual(drains[:4], [("drain", f"node{i}", True) for i in range(4)])
        self.assertEqual(workers[0], ("worker", "node0", "prefill"))
        self.assertEqual(
            workers[1:], [("worker", f"node{i}", "decode") for i in range(1, 4)]
        )
        self.assertTrue(all(item[-1] is True for item in router_roles))
        first_role_change = min(
            actions.index(item) for item in actions if item[0] in ("worker", "router")
        )
        last_initial_drain = max(
            actions.index(item)
            for item in actions
            if item[0] == "drain" and item[-1] is True
        )
        self.assertLess(last_initial_drain, first_role_change)
        self.assertEqual(drains[-4:], [("drain", f"node{i}", False) for i in range(4)])

    def test_request_artifacts_only_claim_flush_after_flush_succeeds(self):
        CandidatePrefillWarmup = self.require_api("CandidatePrefillWarmup")
        with tempfile.TemporaryDirectory() as directory:
            warmup = CandidatePrefillWarmup.__new__(CandidatePrefillWarmup)
            warmup.output_dir = Path(directory)
            (warmup.output_dir / "requests").mkdir()
            results = [
                {
                    "node": "node0",
                    "trace_prompt_kind": "long",
                    "kv_cache_flushed_after": False,
                }
            ]
            warmup._save_json("requests/node0-long.json", results[0])

            warmup._mark_results_flushed(results)

            saved = json.loads(
                (warmup.output_dir / "requests/node0-long.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertTrue(results[0]["kv_cache_flushed_after"])
        self.assertTrue(saved["kv_cache_flushed_after"])

    def test_restore_failure_never_undrains_a_partially_restored_topology(self):
        CandidatePrefillWarmup = self.require_api("CandidatePrefillWarmup")
        NodeSpec = self.require_api("NodeSpec")
        warmup = CandidatePrefillWarmup.__new__(CandidatePrefillWarmup)
        warmup.candidates = [
            NodeSpec(f"node{i}", f"http://n{i}", f"worker-{i}", 8998)
            for i in range(4)
        ]
        warmup.initial_prefill = warmup.candidates[0]
        actions = []

        class Journal:
            def write(self, event, **fields):
                actions.append(("journal", event, fields))

        def drain(node, draining):
            actions.append(("drain", node.name, draining))
            if node.name == "node2" and draining:
                raise RuntimeError("injected drain failure")

        warmup.journal = Journal()
        warmup._set_router_draining = drain
        warmup._set_worker_role = lambda node, role: actions.append(
            ("worker", node.name, role)
        )
        warmup._set_router_role = lambda node, role, draining: actions.append(
            ("router", node.name, role, draining)
        )

        warmup._best_effort_restore()

        self.assertFalse(any(item[0] == "worker" for item in actions))
        self.assertFalse(any(item[0] == "router" for item in actions))
        self.assertFalse(
            any(item[0] == "drain" and item[-1] is False for item in actions)
        )

    def test_run_warms_all_four_candidates_twice_before_flush(self):
        CandidatePrefillWarmup = self.require_api("CandidatePrefillWarmup")
        NodeSpec = self.require_api("NodeSpec")
        warmup = CandidatePrefillWarmup.__new__(CandidatePrefillWarmup)
        warmup.candidates = [
            NodeSpec(f"node{i}", f"http://n{i}", f"worker-{i}", 8998)
            for i in range(4)
        ]
        warmup.nodes = {node.name: node for node in warmup.candidates}
        warmup.initial_prefill = warmup.candidates[0]
        actions = []

        class Client:
            def get_json(self, base_url, path):
                self_path = path
                if self_path != "/pd_flip/router/workers":
                    raise AssertionError(self_path)
                return {
                    "workers": [
                        {
                            "worker_id": f"worker-{i}",
                            "role": "prefill" if i == 0 else "decode",
                            "draining": False,
                        }
                        for i in range(4)
                    ]
                }

        class Journal:
            def write(self, event, **fields):
                actions.append(("journal", event, fields))

        def results(node):
            actions.append(("warm", node.name))
            return [
                {
                    "node": node.name,
                    "trace_prompt_kind": kind,
                    "kv_cache_flushed_after": False,
                }
                for kind in ("long", "short")
            ]

        warmup.client = Client()
        warmup.router_url = "http://router"
        warmup.journal = Journal()
        warmup._warm_initial_prefill = lambda: results(warmup.initial_prefill)
        warmup._warm_decode_candidate = results
        warmup._flush_and_validate = lambda: actions.append(("flush",))

        with tempfile.TemporaryDirectory() as directory:
            warmup.output_dir = Path(directory)
            summary = warmup.run()

        self.assertEqual(summary["warmup_request_count"], 8)
        self.assertEqual(
            [item for item in actions if item[0] == "warm"],
            [("warm", f"node{i}") for i in range(4)],
        )
        self.assertGreater(
            actions.index(("flush",)),
            max(actions.index(("warm", f"node{i}")) for i in range(4)),
        )
        self.assertTrue(
            all(item["kv_cache_flushed_after"] for item in summary["warmup_requests"])
        )


if __name__ == "__main__":
    unittest.main()
