"""CPU-only scheduler checks. No actual process is launched, attached or signaled."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "supervise_amp_training.py"
SPEC = importlib.util.spec_from_file_location("amp_supervisor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fake_checkpoint(path):
    return json.loads(Path(path).read_text())["it"]


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        (self.root / "scripts/train_amp_walk_24gb.sh").touch()
        self.seed = self.root / "seed.pt"
        self.seed.write_text(json.dumps({"it": 10}))
        self.config_path = self.root / "config.json"
        self.config = {
            "rootdir": str(self.root), "max_parallel_trainers": 2,
            "experiments": [{"name": name, "run": f"runs/{name}", "horizon": horizon,
                             "resume_checkpoint": str(self.seed), "target_iterations": 16,
                             "segment_iterations": 2} for name, horizon in (("h8", 8), ("h16", 16), ("queued", 8))],
        }
        self.write_config()
        self.launches = []

    def tearDown(self):
        self.temporary.cleanup()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def launch(self, command, **kwargs):
        child = FakeProcess(9000000 + len(self.launches))
        self.launches.append((command, child))
        return child

    def supervisor(self):
        return MODULE.Supervisor(self.config_path, self.root / "supervisor",
                                 checkpoint_reader=fake_checkpoint, launcher=self.launch,
                                 clock=lambda: 1000.0)

    def complete(self, supervisor, name, iteration, code=0):
        run = self.root / "runs" / name
        run.mkdir(parents=True, exist_ok=True)
        (run / "last.pt").write_text(json.dumps({"it": iteration}))
        supervisor.running[name].child.returncode = code

    def test_segment_scheduler_reload_and_immutable_eval_queue(self):
        supervisor = self.supervisor()
        supervisor.tick()
        self.assertEqual(set(supervisor.running), {"h8", "h16"})
        self.assertEqual(len(self.launches), 2)
        self.assertEqual(MODULE.command_option(self.launches[0][0], "--iterations"), "12")
        self.complete(supervisor, "h8", 12)
        self.config["extra_args"] = ["--action-noise", "0.2"]
        self.write_config()
        supervisor.tick()
        self.assertEqual(supervisor.running["h8"].info["current_target"], 14)
        self.assertIn("--action-noise", self.launches[-1][0])
        queued = self.root / "supervisor/evaluation_checkpoints/h8/ckpt_0000012.pt"
        self.assertEqual(fake_checkpoint(queued), 12)
        # Replacing last.pt, as torch.save+os.replace does, cannot mutate the
        # immutable hard-linked checkpoint already queued for evaluation.
        replacement = self.root / "replacement.pt"
        replacement.write_text(json.dumps({"it": 14}))
        replacement.replace(self.root / "runs/h8/last.pt")
        self.assertEqual(fake_checkpoint(queued), 12)
        self.complete(supervisor, "h8", 14)
        supervisor.tick()
        self.assertEqual(supervisor.running["h8"].info["current_target"], 16)
        self.complete(supervisor, "h8", 16)
        supervisor.tick()
        self.assertEqual(supervisor.states["h8"]["state"], "completed")
        self.assertIn("queued", supervisor.running)
        jobs = [json.loads(line) for line in (self.root / "supervisor/evaluation_jobs.jsonl").read_text().splitlines()]
        self.assertEqual([job["checkpoint_iteration"] for job in jobs], [12, 14, 16])

    def test_failure_stays_review_until_new_ack(self):
        supervisor = self.supervisor()
        supervisor.tick()
        self.complete(supervisor, "h8", 11, code=1)
        supervisor.tick()
        self.assertEqual(supervisor.states["h8"]["state"], "needs_review")
        before = len(self.launches)
        supervisor.tick()
        self.assertEqual(len(self.launches), before)
        # Free the other slots, then explicitly acknowledge a repaired branch.
        self.complete(supervisor, "h16", 16)
        self.complete(supervisor, "queued", 16)
        self.config["experiments"][0]["review_ack"] = "inspected-and-fixed"
        self.write_config()
        supervisor.tick()
        self.assertEqual(supervisor.states["h8"]["state"], "running")
        self.assertEqual(supervisor.running["h8"].info["current_target"], 13)

    def test_health_checks_and_partial_log(self):
        run = self.root / "runs/h8"
        run.mkdir(parents=True)
        log = run / "log.jsonl"
        log.write_text('{"it": 12, "wm_errk": NaN}\n{"it":13')
        stdout = self.root / "stdout.log"
        stdout.touch()
        process = {"stdout": str(stdout), "start_iteration": 10, "started_at_epoch": 1000}
        experiment = {"run": str(run)}
        self.assertIn("nonfinite", MODULE.health_problem(experiment, process, 1000, 900))
        log.write_text('{"it": 12, "wm_errk": 0.1}\n')
        stdout.write_text("torch.OutOfMemoryError: CUDA out of memory")
        self.assertIn("fatal_output", MODULE.health_problem(experiment, process, 1000, 900))
        stdout.write_text("")
        import os
        os.utime(log, (1000, 1000))
        self.assertIn("stale_log", MODULE.health_problem(experiment, process, 2000, 900))

    def test_successful_dead_adopted_process_requires_target_checkpoint(self):
        supervisor = self.supervisor()
        run = self.root / "runs/h8"
        run.mkdir(parents=True)
        (run / "last.pt").write_text(json.dumps({"it": 12}))
        active = self.root / "active.json"
        active.write_text(json.dumps({"processes": [{"pid": 9999999, "out": str(run),
            "horizon": 8, "stdout": str(self.root / "missing.log"),
            "command": ["python", "-m", "supermimic.amp_train", "--iterations", "12", "--horizon", "8"]}]}))
        supervisor.adopt(active)
        self.assertFalse(supervisor.running["h8"].alive())
        supervisor.tick()
        self.assertEqual(supervisor.running["h8"].info["current_target"], 14)

    def test_config_guardrails_and_segment_caps(self):
        self.assertEqual(MODULE.next_segment(12000, 50000, 2000), 14000)
        self.assertEqual(MODULE.next_segment(49000, 50000, 2000), 50000)
        self.assertIsNone(MODULE.next_segment(50000, 50000, 2000))
        self.config["max_parallel_trainers"] = 3
        self.write_config()
        with self.assertRaises(ValueError):
            MODULE.load_config(self.config_path)
        with self.assertRaises(ValueError):
            MODULE.arguments(["--out=wrong"])

    def test_shutdown_accepts_gracefully_saved_partial_segment(self):
        supervisor = self.supervisor()
        supervisor.tick()
        for name, process in supervisor.running.items():
            def interrupt(name=name, process=process):
                (self.root / "runs" / name / "last.pt").write_text(json.dumps({"it": 11}))
                process.child.returncode = 0
            process.interrupt = interrupt
        supervisor.shutdown(timeout=1)
        self.assertFalse(supervisor.running)
        self.assertTrue(all(supervisor.states[name]["state"] == "stopped" for name in ("h8", "h16")))
        self.assertFalse((self.root / "supervisor/evaluation_jobs.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
