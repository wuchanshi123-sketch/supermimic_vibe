"""Continue bounded AMP training segments, preserving checkpoints for review.

Run with the project's Python:
  .venv/bin/python code/scripts/supervise_amp_training.py CONFIG.json \
    --adopt runs/amp_supervision/active.json

Config example:
  {"rootdir": "/path/to/supermimic_vibe", "max_parallel_trainers": 2,
   "extra_args": [], "experiments": [
     {"name": "h8", "run": "runs/amp_walk_h8_control", "horizon": 8,
      "resume_checkpoint": "runs/amp_walk_h8_control/last.pt",
      "target_iterations": 50000, "segment_iterations": 2000}]}

The config is reread every poll; targets and arguments apply at the next segment.
Failed branches remain needs_review. To resume one after inspecting/fixing it,
change its optional review_ack string to a new value. No algorithm is selected
or changed by this script. Evaluation jobs are queued, not automatically run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


RESERVED = {"--out", "--resume", "--iterations", "--horizon"}
FATAL_OUTPUT = re.compile(r"CUDA out of memory|OutOfMemoryError|FloatingPointError|"
                          r"Non-finite|non-finite|Traceback \(most recent call last\)", re.I)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def resolve(root, path):
    path = Path(path).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def arguments(value):
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("extra_args must be an array of strings")
    if any(item.split("=", 1)[0] in RESERVED for item in value):
        raise ValueError("extra_args cannot override out, resume, iterations or horizon")
    return value


def load_config(path):
    raw = json.loads(path.read_text())
    root = Path(raw["rootdir"]).expanduser().resolve()
    if not (root / "scripts/train_amp_walk_24gb.sh").is_file():
        raise ValueError(f"AMP launcher missing under rootdir {root}")
    maximum = positive_integer(raw.get("max_parallel_trainers", 2), "max_parallel_trainers")
    if maximum > 2:
        raise ValueError("max_parallel_trainers cannot exceed two")
    shared = arguments(raw.get("extra_args", []))
    experiments = {}
    runs = set()
    for source in raw["experiments"]:
        experiment = dict(source)
        name = experiment["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("experiment names must contain letters, numbers, dot, dash or underscore")
        if name in experiments:
            raise ValueError(f"duplicate experiment {name}")
        run = resolve(root, experiment["run"])
        if run in runs:
            raise ValueError(f"multiple experiments cannot write the same run: {run}")
        runs.add(run)
        experiment.update(
            name=name, run=str(run),
            horizon=positive_integer(experiment["horizon"], "horizon"),
            target_iterations=positive_integer(experiment.get("target_iterations", 50000), "target_iterations"),
            segment_iterations=positive_integer(experiment.get("segment_iterations", 2000), "segment_iterations"),
            resume_checkpoint=str(resolve(root, experiment.get("resume_checkpoint", run / "last.pt"))),
            extra_args=shared + arguments(experiment.get("extra_args", [])),
            review_ack=str(experiment.get("review_ack", "")),
        )
        experiments[name] = experiment
    if not experiments:
        raise ValueError("at least one experiment is required")
    return {"rootdir": str(root), "max_parallel_trainers": maximum, "experiments": experiments}


def next_segment(iteration, target, segment):
    return min(target, iteration + segment) if iteration < target else None


def trainer_command(experiment, checkpoint, target):
    return ["bash", "scripts/train_amp_walk_24gb.sh", "train",
            *experiment["extra_args"], "--resume", str(checkpoint),
            "--out", experiment["run"], "--iterations", str(target),
            "--horizon", str(experiment["horizon"])]


def checkpoint_iteration(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != "supermimic_vibe_amp_from_scratch_v1":
        raise ValueError(f"unexpected checkpoint format: {path}")
    iteration = checkpoint.get("it")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"invalid checkpoint iteration: {path}")
    return iteration


def process_identity(pid):
    """Linux start ticks prevent signaling an unrelated process after PID reuse."""
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if tail[0] == "Z" else tail[19]
    except (OSError, IndexError):
        return None


def command_option(command, option):
    values = [command[index + 1] for index, item in enumerate(command[:-1]) if item == option]
    values += [item.split("=", 1)[1] for item in command if item.startswith(option + "=")]
    return values[-1] if values else None


def existing_trainers(root, run):
    matches = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            command = (directory / "cmdline").read_bytes().decode().split("\0")
            output = command_option(command, "--out")
            if "supermimic.amp_train" in command and output and resolve(root, output) == run:
                matches.append(int(directory.name))
        except (OSError, UnicodeDecodeError):
            pass
    return matches


def tail_text(path, size=65536):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode(errors="replace")
    except OSError:
        return ""


def latest_log(path):
    for line in reversed(tail_text(path).splitlines()):
        try:
            row = json.loads(line)
            if isinstance(row, dict) and "it" in row:
                return row
        except json.JSONDecodeError:
            pass
    return None


def health_problem(experiment, process, now, stale_seconds):
    output = tail_text(process["stdout"])
    match = FATAL_OUTPUT.search(output)
    if match:
        return "fatal_output: " + match.group(0)
    log = Path(experiment["run"]) / "log.jsonl"
    row = latest_log(log)
    if row and row.get("it", -1) >= process.get("start_iteration", 0):
        bad = [key for key, value in row.items()
               if isinstance(value, (int, float)) and not math.isfinite(value)]
        if bad:
            return "nonfinite_metrics: " + ", ".join(bad)
    last_write = max(process["started_at_epoch"], log.stat().st_mtime if log.exists() else 0)
    if now - last_write > stale_seconds:
        return f"stale_log: no progress write for {now - last_write:.0f}s"
    return None


@dataclass
class Running:
    info: dict
    child: object = None
    stopping_reason: str | None = None

    def alive(self):
        if self.child is not None:
            return self.child.poll() is None
        return process_identity(self.info["pid"]) == self.info.get("process_start_ticks") and self.info.get("process_start_ticks") is not None

    def interrupt(self):
        if self.alive():
            os.kill(self.info["pid"], signal.SIGINT)


class Supervisor:
    def __init__(self, config_path, out, stale_seconds=900, checkpoint_reader=checkpoint_iteration,
                 launcher=subprocess.Popen, clock=time.time):
        self.config_path, self.out = config_path, out
        self.config = load_config(config_path)
        self.root = Path(self.config["rootdir"])
        self.checkpoint_reader, self.launcher, self.clock = checkpoint_reader, launcher, clock
        self.stale_seconds = stale_seconds
        self.running = {}
        self.states = {}
        self.queued = set()
        self.config_error = None
        self.shutdown_requested = False
        self.out.mkdir(parents=True, exist_ok=True)
        queue = self.out / "evaluation_jobs.jsonl"
        if queue.exists():
            for line in queue.read_text().splitlines():
                try:
                    self.queued.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        status = self.out / "status.json"
        if status.exists():
            try:
                self.states = json.loads(status.read_text()).get("branches", {})
            except json.JSONDecodeError:
                pass
        for name, experiment in self.config["experiments"].items():
            self.states.setdefault(name, {"state": "pending", "review_ack": experiment["review_ack"]})

    def event(self, event, **fields):
        record = {"time": timestamp(), "event": event, **fields}
        with (self.out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        print(json.dumps(record), flush=True)

    def reload(self):
        try:
            updated = load_config(self.config_path)
            if updated["rootdir"] != str(self.root):
                raise ValueError("rootdir cannot change while supervisor is running")
            for name, process in self.running.items():
                previous = self.config["experiments"].get(name, process.info)
                current = updated["experiments"].get(name, previous)
                if any(previous[key] != current[key] for key in ("run", "horizon")):
                    raise ValueError(f"cannot change run/horizon while {name} has an active segment")
            if updated != self.config:
                self.event("config_reloaded")
                self.config = updated
            self.config_error = None
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            message = str(exc)
            if message != self.config_error:
                self.event("config_invalid", reason=message)
            self.config_error = message
        for name, experiment in self.config["experiments"].items():
            state = self.states.setdefault(name, {"state": "pending", "review_ack": experiment["review_ack"]})
            if state.get("state") == "needs_review" and experiment["review_ack"] != state.get("review_ack", ""):
                state.update(state="pending", reason=None, review_ack=experiment["review_ack"])
                self.event("review_acknowledged", name=name, review_ack=experiment["review_ack"])

    def adopt(self, source):
        data = json.loads(source.read_text())
        entries = data.get("processes", [data] if "pid" in data else [])
        for entry in entries:
            command = entry.get("command", [])
            run = entry.get("out", entry.get("run", command_option(command, "--out")))
            if run is None:
                self.event("adoption_skipped", reason="entry has no output run", entry=entry)
                continue
            run = resolve(self.root, run)
            candidates = [experiment for experiment in self.config["experiments"].values()
                          if Path(experiment["run"]) == run]
            if not candidates:
                self.event("adoption_skipped", reason="run absent from config", run=str(run))
                continue
            experiment = candidates[0]
            name = experiment["name"]
            if name in self.running:
                continue
            pid = positive_integer(entry["pid"], "pid")
            identity = process_identity(pid)
            if identity is not None:
                try:
                    actual = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
                except (OSError, UnicodeDecodeError):
                    actual = []
                if process_identity(pid) is None:
                    # The segment can finish between the identity and cmdline
                    # reads; its target checkpoint is still verifiable below.
                    identity = None
                else:
                    actual_run = command_option(actual, "--out")
                    if "supermimic.amp_train" not in actual or (actual_run and resolve(self.root, actual_run) != run):
                        self.event("adoption_rejected", name=name, pid=pid, reason="PID is not the configured AMP trainer")
                        self.states[name].update(state="needs_review", reason="adoption PID mismatch")
                        continue
                    command = actual
            target = command_option(command, "--iterations") or entry.get("current_target")
            if target is None:
                self.event("adoption_rejected", name=name, pid=pid, reason="unknown segment target")
                self.states[name].update(state="needs_review", reason="unknown adopted segment target")
                continue
            adopted_horizon = command_option(command, "--horizon") or entry.get("horizon")
            if adopted_horizon is not None and int(adopted_horizon) != experiment["horizon"]:
                self.states[name].update(state="needs_review", reason="adopted horizon differs from config")
                self.event("adoption_rejected", name=name, pid=pid, reason="horizon mismatch")
                continue
            start = entry.get("started_at_epoch", self.clock())
            log = run / "log.jsonl"
            row = latest_log(log)
            info = {
                "pid": pid, "process_start_ticks": identity, "adopted": True,
                "current_target": int(target), "started_at_epoch": start,
                "start_iteration": entry.get("start_iteration", row.get("it", 0) if row else 0),
                "stdout": str(resolve(self.root, entry.get("stdout", self.out / f"adopted_{pid}.log"))),
                "command": command, "run": str(run), "horizon": experiment["horizon"],
            }
            self.running[name] = Running(info)
            self.states[name].update(state="running", process=info)
            self.event("adopted", name=name, **info)

    def adopt_saved_state(self):
        """Restarting the supervisor can recover its own previously tracked PIDs."""
        saved = [state["process"] for state in self.states.values()
                 if state.get("state") in ("running", "stopping") and "process" in state]
        if saved:
            temporary = self.out / "adopt_saved.json"
            temporary.write_text(json.dumps({"processes": saved}))
            self.adopt(temporary)

    def require_review(self, name, reason):
        experiment = self.config["experiments"].get(name, {})
        self.states[name].update(state="needs_review", reason=reason,
                                 review_ack=experiment.get("review_ack", self.states[name].get("review_ack", "")))
        self.event("needs_review", name=name, reason=reason)

    def queue_evaluation(self, name, experiment, iteration):
        source = Path(experiment["run"]) / "last.pt"
        destination = self.out / "evaluation_checkpoints" / name / f"ckpt_{iteration:07d}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                # Exclusive creation avoids replacing a previously queued file.
                with source.open("rb") as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
        if self.checkpoint_reader(destination) != iteration:
            raise ValueError(f"queued checkpoint iteration mismatch: {destination}")
        identifier = f"{name}:{iteration}"
        if identifier not in self.queued:
            digest = hashlib.sha256()
            with destination.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            job = {"id": identifier, "created_at": timestamp(), "status": "pending",
                   "name": name, "run": experiment["run"], "horizon": experiment["horizon"],
                   "checkpoint_iteration": iteration, "checkpoint": str(destination),
                   "sha256": digest.hexdigest(), "reason": "segment_boundary",
                   "scope": "Queue for fixed-command gait/style evaluation; training completion is not convergence."}
            with (self.out / "evaluation_jobs.jsonl").open("a") as stream:
                stream.write(json.dumps(job) + "\n")
                stream.flush()
            self.queued.add(identifier)
            self.event("evaluation_queued", name=name, checkpoint=str(destination), iteration=iteration)

    def finish(self, name, process):
        experiment = self.config["experiments"].get(name)
        if experiment is None:
            experiment = {"run": process.info["run"], "horizon": process.info["horizon"]}
        state = self.states[name]
        code = process.child.poll() if process.child is not None else None
        shutdown_stop = process.stopping_reason == "supervisor_shutdown_interrupt"
        reason = None if shutdown_stop else process.stopping_reason
        if reason is None and code not in (None, 0):
            reason = f"trainer_exit_code: {code}"
        if reason is None:
            match = FATAL_OUTPUT.search(tail_text(process.info["stdout"]))
            if match:
                reason = "fatal_output: " + match.group(0)
        try:
            iteration = self.checkpoint_reader(Path(experiment["run"]) / "last.pt")
            state["last_iteration"] = iteration
            if reason is None and not shutdown_stop and iteration < process.info["current_target"]:
                reason = f"checkpoint_short_of_segment_target: {iteration} < {process.info['current_target']}"
            if reason is None and iteration >= process.info["current_target"]:
                self.queue_evaluation(name, experiment, iteration)
        except Exception as exc:
            reason = reason or f"checkpoint_verification_failed: {exc}"
        state.pop("process", None)
        del self.running[name]
        if reason:
            self.require_review(name, reason)
        elif self.shutdown_requested:
            state.update(state="stopped", reason="supervisor shutdown")
        else:
            state.update(state="pending", reason=None)
            self.event("segment_completed", name=name, iteration=state["last_iteration"],
                       adopted_exit_status_unavailable=process.child is None)

    def launch(self, name, experiment):
        run = Path(experiment["run"])
        checkpoint = run / "last.pt"
        if not checkpoint.exists():
            checkpoint = Path(experiment["resume_checkpoint"])
        try:
            unadopted = existing_trainers(self.root, run)
            if unadopted:
                raise RuntimeError(f"existing unadopted trainer PIDs {unadopted}; use --adopt")
            iteration = self.checkpoint_reader(checkpoint)
            target = next_segment(iteration, experiment["target_iterations"], experiment["segment_iterations"])
            if target is None:
                self.states[name].update(state="completed", last_iteration=iteration, reason="configured iteration target reached; quality is not certified")
                self.event("iteration_target_reached", name=name, iteration=iteration)
                return
            run.mkdir(parents=True, exist_ok=True)
            command = trainer_command(experiment, checkpoint, target)
            output_dir = self.out / "stdout"
            output_dir.mkdir(exist_ok=True)
            stdout = output_dir / f"{name}_{iteration:07d}_to_{target:07d}_{time.time_ns()}.log"
            with stdout.open("ab") as stream:
                child = self.launcher(command, cwd=self.root, stdout=stream,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            info = {"pid": child.pid, "process_start_ticks": process_identity(child.pid),
                    "adopted": False, "current_target": target,
                    "started_at_epoch": self.clock(), "start_iteration": iteration,
                    "stdout": str(stdout), "command": command,
                    "run": experiment["run"], "horizon": experiment["horizon"]}
            self.running[name] = Running(info, child)
            self.states[name].update(state="running", process=info, last_iteration=iteration,
                                     review_ack=experiment["review_ack"], reason=None)
            self.event("segment_started", name=name, **info)
        except Exception as exc:
            self.require_review(name, f"launch_failed: {exc}")

    def tick(self):
        self.reload()
        for name, process in list(self.running.items()):
            if not process.alive():
                self.finish(name, process)
                continue
            experiment = self.config["experiments"].get(name, {"run": process.info["run"]})
            reason = health_problem(experiment, process.info, self.clock(), self.stale_seconds)
            if reason and process.stopping_reason is None:
                process.stopping_reason = reason
                self.states[name].update(state="stopping", reason=reason)
                self.event("trainer_interrupt_requested", name=name, reason=reason)
                process.interrupt()
        if not self.shutdown_requested and self.config_error is None:
            for name, experiment in self.config["experiments"].items():
                state = self.states[name]
                if name in self.running or state.get("state") == "needs_review":
                    continue
                # A larger target in edited config reactivates an earlier
                # completed branch without discarding its learning state.
                if state.get("state") == "completed" and state.get("last_iteration", 0) >= experiment["target_iterations"]:
                    continue
                if len(self.running) >= self.config["max_parallel_trainers"]:
                    break
                self.launch(name, experiment)
        self.write_status()

    def write_status(self):
        report = {"updated_at": timestamp(), "supervisor_pid": os.getpid(),
                  "config": str(self.config_path), "config_error": self.config_error,
                  "shutdown_requested": self.shutdown_requested, "running_trainers": len(self.running),
                  "max_parallel_trainers": self.config["max_parallel_trainers"],
                  "branches": self.states, "evaluation_jobs_queued": len(self.queued),
                  "scope": "Durable segmented continuation, not walking convergence certification; failures require review."}
        temporary = self.out / "status.tmp.json"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.out / "status.json")

    def shutdown(self, timeout):
        self.shutdown_requested = True
        self.event("supervisor_shutdown_requested")
        for name, process in self.running.items():
            process.stopping_reason = "supervisor_shutdown_interrupt"
            process.interrupt()
        deadline = self.clock() + timeout
        while self.running and self.clock() < deadline:
            for name, process in list(self.running.items()):
                if not process.alive():
                    self.finish(name, process)
            self.write_status()
            if self.running:
                time.sleep(1)
        if self.running:
            self.event("shutdown_save_wait_timed_out", remaining_pids=[process.info["pid"] for process in self.running.values()])
        self.write_status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, help="status/events/evaluation queue directory; default CONFIG-parent/supervisor")
    parser.add_argument("--adopt", nargs="?", const="runs/amp_supervision/active.json", type=Path)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--stale-seconds", type=float, default=900)
    parser.add_argument("--shutdown-timeout", type=float, default=180)
    parser.add_argument("--dry-run", action="store_true", help="print configuration/planned segments; never attach, start, signal, or write files")
    args = parser.parse_args()
    if min(args.interval, args.stale_seconds, args.shutdown_timeout) <= 0:
        parser.error("interval, stale-seconds and shutdown-timeout must be positive")
    config_path = args.config.resolve()
    try:
        config = load_config(config_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    out = args.out.resolve() if args.out else config_path.parent / "supervisor"
    if args.dry_run:
        plans = []
        for experiment in config["experiments"].values():
            checkpoint = Path(experiment["run"]) / "last.pt"
            if not checkpoint.exists():
                checkpoint = Path(experiment["resume_checkpoint"])
            try:
                iteration = checkpoint_iteration(checkpoint)
                target = next_segment(iteration, experiment["target_iterations"], experiment["segment_iterations"])
                plans.append({"name": experiment["name"], "resume_iteration": iteration,
                              "next_target": target, "command": trainer_command(experiment, checkpoint, target) if target else None})
            except Exception as exc:
                plans.append({"name": experiment["name"], "error": str(exc)})
        print(json.dumps({"dry_run": True, "adoption_not_performed": True, "config": config, "plans": plans}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    lock_path = Path(config["rootdir"]) / "runs/amp_supervision/supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"another supervisor holds {lock_path}")
        supervisor = Supervisor(config_path, out, args.stale_seconds)
        supervisor.adopt_saved_state()
        if args.adopt:
            supervisor.adopt(resolve(supervisor.root, args.adopt))
        supervisor.event("supervisor_started", config=str(config_path), pid=os.getpid())
        def request_shutdown(signum, frame):
            supervisor.shutdown_requested = True
        signal.signal(signal.SIGINT, request_shutdown)
        signal.signal(signal.SIGTERM, request_shutdown)
        try:
            while not supervisor.shutdown_requested:
                supervisor.tick()
                time.sleep(args.interval)
        except Exception as exc:
            supervisor.event("supervisor_error", reason=repr(exc))
            supervisor.shutdown_requested = True
            raise
        finally:
            supervisor.shutdown(args.shutdown_timeout)


if __name__ == "__main__":
    main()
