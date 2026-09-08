"""End-to-end test of the MA-Crafter condition runner with a scripted LLM client.

Runs a short episode for every topology through experiment/run_condition.py
and checks the standard result files. No API credentials are needed. Run
from the environment directory:

    python -m unittest tests.test_run_condition
"""

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # figures are written to files, never shown

from cognitive.agent import InterruptDecision
from experiment.run_condition import TOPOLOGY_BUILDERS, run_condition
from tests.test_interrupt_decision import ScriptedLLMClient

EXPECTED_FILES = ("plan_logs.json", "llm_usage.json", "coop2_metrics.json", "coop2_metrics.csv")


def _args(topology: str, output_root: str, **overrides) -> argparse.Namespace:
    values = dict(
        topology=topology, agents=3, steps=6, time_limit_seconds=120, seed=42, model=None, backend=None,
        goal="", coop2_repair=False, quiet=True, llm_quiet=True, show=False, record_video=False,
        output_root=Path(output_root),
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class RunConditionTests(unittest.TestCase):
    def _run(self, topology: str, **overrides) -> Path:
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                output_dir = run_condition(
                    _args(topology, tmp, **overrides),
                    llm_client=ScriptedLLMClient(InterruptDecision.RESUME),
                )
            for name in EXPECTED_FILES:
                self.assertTrue((output_dir / name).exists(), f"{topology}: missing {name}")
            return json.loads((output_dir / "llm_usage.json").read_text())

    def test_every_topology_writes_standard_outputs(self):
        for topology in TOPOLOGY_BUILDERS:
            with self.subTest(topology=topology):
                usage = self._run(topology)
                self.assertEqual(usage["topology"], topology)
                self.assertEqual(set(usage["per_agent"]), {"agent_0", "agent_1", "agent_2"})
                self.assertGreater(sum(a["api_calls"] for a in usage["per_agent"].values()), 0)

    def test_roles_are_labeled_per_topology(self):
        roles = {r["role"] for r in self._run("centralized")["per_agent"].values()}
        self.assertEqual(roles, {"leader", "follower"})
        roles = {r["role"] for r in self._run("broadcast_chain")["per_agent"].values()}
        self.assertEqual(roles, {"speaker-0", "speaker-1", "speaker-2"})

    def test_repair_condition_runs(self):
        self._run("centralized", coop2_repair=True)


if __name__ == "__main__":
    unittest.main()
