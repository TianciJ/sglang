import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "pd_flip_qwen80b_16_instance.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pd_flip_qwen80b_16_instance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Qwen80B16InstanceRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def build_runner(self):
        handle = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.addCleanup(lambda: os.unlink(handle.name))
        handle.write("ADMIN_API_KEY=local-test-secret\n")
        handle.close()
        return self.module.Runner(handle.name, "test-16-instance")

    def test_builds_four_hosts_and_sixteen_unique_instances(self):
        runner = self.build_runner()

        self.assertEqual(len(runner.instances), 16)
        self.assertEqual(len({item["name"] for item in runner.instances}), 16)
        self.assertEqual(len({runner.worker_name(item) for item in runner.instances}), 16)
        self.assertEqual(
            [item["role"] for item in runner.instances].count("prefill"), 1
        )
        self.assertEqual(
            [item["role"] for item in runner.instances].count("decode"), 15
        )
        for host_index in range(4):
            host_instances = [
                item for item in runner.instances if item["host_index"] == host_index
            ]
            self.assertEqual(
                [item["gpu_ids"] for item in host_instances],
                ["0,1", "2,3", "4,5", "6,7"],
            )

    def test_uses_single_coordinated_flip(self):
        runner = self.build_runner()

        self.assertEqual(runner.source_name, "h2i2")
        self.assertEqual(runner.target_name, "h3i3")
        self.assertEqual(len(runner.node_args()) // 2, 16)

    def test_completed_observer_without_trigger_stops_exact_controller(self):
        runner = self.build_runner()
        with tempfile.TemporaryDirectory() as directory:
            runner.run_dir = directory
            (Path(directory) / "observer").mkdir()
            (Path(directory) / "status").mkdir()
            (Path(directory) / "logs").mkdir()
            (Path(directory) / "observer" / "summary.json").write_text(
                '{"first_trigger": null}\n', encoding="utf-8"
            )
            commands = []
            runner.coordinator = lambda command, **kwargs: commands.append(command)

            with self.assertRaisesRegex(
                RuntimeError, "observer completed without an SLO trigger"
            ):
                runner.require_observer_trigger()

            invalid = (Path(directory) / "status" / "invalid.json").read_text()
            self.assertIn("observer_completed_without_slo_trigger", invalid)
            self.assertEqual(len(commands), 1)
            self.assertIn(runner.helper_name("controller"), commands[0])
            self.assertIn("docker stop --time 60", commands[0])

    def test_runner_has_no_forbidden_or_broad_cleanup(self):
        source = SCRIPT.read_text(encoding="utf-8")

        for forbidden in (
            "docker restart",
            "docker rm -f",
            "docker kill",
            "pkill",
            "killall",
            "kill -9",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("self.run_id", source)
        self.assertIn("observer completed without an SLO trigger", source)
        self.assertIn("1P15D", source)
        self.assertIn("2P14D", source)


if __name__ == "__main__":
    unittest.main()
