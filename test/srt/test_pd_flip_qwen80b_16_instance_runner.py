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

    def test_runner_has_no_forbidden_or_broad_cleanup(self):
        source = SCRIPT.read_text(encoding="utf-8")

        for forbidden in (
            "docker restart",
            "docker rm -f",
            "pkill",
            "killall",
            "kill -9",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("self.run_id", source)
        self.assertIn("docker kill --signal=INT", source)
        self.assertIn("1P15D", source)
        self.assertIn("2P14D", source)


if __name__ == "__main__":
    unittest.main()
