from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class EvaluationSupervisorTests(unittest.TestCase):
    def test_service_launches_one_yaml_on_both_gpus(self):
        wrapper = (REPO / "evaluation/supervisor/opd32b-eval.sh").read_text()
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", wrapper)
        self.assertIn(
            "--config evaluation/configs/nemotron_cascade2.yaml",
            wrapper,
        )
        for hidden_override in ("MODEL_MODE", "DFLASH=", "MAXREQ=", "CTX="):
            self.assertNotIn(hidden_override, wrapper)

    def test_supervisor_owns_the_full_process_group(self):
        config = (REPO / "evaluation/supervisor/opd32b-eval.conf").read_text()
        self.assertIn("autostart=false", config)
        self.assertIn("autorestart=unexpected", config)
        self.assertIn("stopasgroup=true", config)
        self.assertIn("killasgroup=true", config)
        self.assertIn("stdout_logfile=/dev/stdout", config)

    def test_dry_run_is_supervised_and_never_restarts_api_calls(self):
        wrapper = (
            REPO / "evaluation/supervisor/imo-2025-problem1-eval.sh"
        ).read_text()
        config = (
            REPO / "evaluation/supervisor/imo-2025-problem1-eval.conf"
        ).read_text()
        self.assertIn("run_full_evaluation.py", wrapper)
        self.assertIn("imo-2025-problem-1.json", wrapper)
        self.assertIn("imo-2025-problem-1-dryrun-20260712", wrapper)
        self.assertIn("autorestart=false", config)


if __name__ == "__main__":
    unittest.main()
