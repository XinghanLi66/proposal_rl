"""
BenchmarkPipeline: orchestrates proposal generation → worker → eval.

All intermediate state is written to disk so the TUI can monitor it live.

Directory layout per run:
  run_dir/
    config.json                  ← BenchmarkConfig as JSON
    sample_{i:02d}/
      state.json                 ← {"step": ..., "improvement": ..., "passed": ..., "elapsed_s": ...}
      prompt_record.json         ← record fed to proposal model (refs, system, prompt)
      proposal.txt               ← raw model output
      worker_prompt.txt          ← full prompt given to Claude worker
      worker.log                 ← Claude Code CLI stdout (streaming)
      editable_region.py         ← worker's implementation (written by worker)
      eval.log                   ← MLS-Bench eval stdout (streaming)
      result.json                ← {"val_metric": x, "_sig": y}
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from benchmark.tasks.base import AbstractBenchmarkTask

# Module-level model cache so we don't reload the 7B model per sample
_model_cache: dict = {}
_model_lock = threading.Lock()

# Per-(machine, gpu) lock: ensures only one generation subprocess runs per GPU.
# This prevents OOM when multiple sweep pipelines share the same generation GPU.
_gen_gpu_locks: dict[tuple[str, str], threading.Lock] = {}
_gen_gpu_locks_mutex = threading.Lock()


def _get_gen_lock(machine: str, gpu: str) -> threading.Lock:
    key = (machine, gpu)
    with _gen_gpu_locks_mutex:
        if key not in _gen_gpu_locks:
            _gen_gpu_locks[key] = threading.Lock()
        return _gen_gpu_locks[key]


def _free_vram_mb(machine: str, gpu: str) -> int:
    """Return free VRAM in MB on (machine, gpu). Returns 0 on failure."""
    cmd = (
        f"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits "
        f"--id={gpu}"
    )
    try:
        if machine and not _is_local(machine):
            result = subprocess.run(
                ["ssh", machine, cmd], capture_output=True, text=True, timeout=10
            )
        else:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
        lines = result.stdout.strip().splitlines()
        if lines:
            return int(lines[0].strip())
    except Exception:
        pass
    return 0


# Minimum free VRAM (MB) required before starting a generation subprocess.
# A 7B model in bf16 needs ~14GB; leave 2GB headroom → 16GB minimum.
_GEN_VRAM_MIN_MB = 16_000
_GEN_VRAM_POLL_S = 30  # seconds between VRAM checks

_WORKER_PROMPT_TEMPLATE = (Path(__file__).parent / "mls_worker_prompt.txt").read_text()
WORKER_TIMEOUT = 7200  # 2 hours per worker (200-epoch MLS-Bench ResNet takes ~60-90 min)

# Env vars forwarded to remote SSH workers
_FORWARDED_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "BENCHMARK_HMAC_KEY",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
]


def _is_local(machine: str) -> bool:
    """Return True if the machine string refers to the current host or is empty."""
    if not machine:
        return True
    import socket
    hostname = socket.gethostname()
    # Map SSH config aliases to check if we're the same host
    _ALIAS_MAP = {
        "lxh_agent_0": "lxh_agent_0",
        "lxh_agent_1": "lxh_agent_1",
        "lxh_agent_2": "lxh_agent_2",
        "lxh_agent_3": "lxh_agent_3",
    }
    # Resolve current host alias from SSH config by comparing to known patterns
    for alias, _ in _ALIAS_MAP.items():
        # If SSH config has HostName matching our hostname, consider it local
        ssh_config = Path.home() / ".ssh" / "config"
        if ssh_config.exists():
            text = ssh_config.read_text()
            # Parse: find Host block for alias and extract HostName
            lines = text.splitlines()
            in_host = False
            for line in lines:
                stripped = line.strip()
                if stripped.lower().startswith("host ") and stripped.split()[1] == alias:
                    in_host = True
                elif stripped.lower().startswith("host "):
                    in_host = False
                elif in_host and stripped.lower().startswith("hostname"):
                    remote_host = stripped.split()[-1]
                    if remote_host == hostname or hostname.startswith(remote_host.split(".")[0]):
                        if machine == alias:
                            return True
    return machine == hostname


def _build_ssh_cmd(machine: str, remote_cmd: list[str], env: dict | None = None) -> list[str]:
    """Wrap remote_cmd in an SSH call, forwarding selected env vars."""
    env_prefix = ""
    if env:
        pairs = []
        for k in _FORWARDED_ENV_VARS:
            if k in env:
                v = env[k].replace("'", "'\\''")
                pairs.append(f"{k}='{v}'")
        if pairs:
            env_prefix = " ".join(pairs) + " "
    escaped = " ".join(f"'{a}'" for a in remote_cmd)
    return ["ssh", machine, f"{env_prefix}{escaped}"]


class PipelineStep(str, Enum):
    PENDING           = "pending"
    BUILDING_PROMPT   = "building_prompt"
    GENERATING        = "generating"
    RUNNING_WORKER    = "running_worker"
    RUNNING_EVAL      = "running_eval"   # eval embedded in worker run.sh; this state unused for now
    DONE              = "done"
    ERROR             = "error"

    def label(self) -> str:
        return {
            "pending":         "Pending",
            "building_prompt": "Building prompt",
            "generating":      "Generating proposal",
            "running_worker":  "Worker running",
            "running_eval":    "MLS eval running",
            "done":            "Done",
            "error":           "Error",
        }[self.value]


@dataclass
class SampleState:
    index: int
    step: PipelineStep = PipelineStep.PENDING
    improvement: float | None = None
    passed: bool | None = None
    elapsed_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["step"] = self.step.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SampleState":
        d = dict(d)
        d["step"] = PipelineStep(d.get("step", "pending"))
        return cls(**d)

    def save(self, sample_dir: Path) -> None:
        (sample_dir / "state.json").write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, sample_dir: Path) -> "SampleState | None":
        p = sample_dir / "state.json"
        if not p.exists():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except Exception:
            return None


@dataclass
class BenchmarkConfig:
    task_name: str          # e.g. "dl_lr_schedule"
    checkpoint: str | None  # path to checkpoint dir, or None for API
    strategy: str           # prompt builder strategy
    n_samples: int          # number of proposals to generate
    run_id: str             # unique run identifier
    gpu_device: str = "0"   # CUDA device for eval
    max_workers: int = 2    # parallel worker count
    claude_cmd: str = "/newcpfs/lxh/claude-home-agent1/run_claude.sh"
    conda_env: str | None = None
    is_api: bool = False    # True = use Claude API, not checkpoint
    api_model: str = "claude-opus-4-6"
    run_dir: str = ""       # filled in at runtime
    subtask: str = ""       # e.g. "resnet20-cifar10"; empty = task default
    machine: str = ""       # SSH host for remote execution; "" = local
    temperature: float = 0.7 # checkpoint sampling temperature

    def save(self, run_dir: Path) -> None:
        (run_dir / "config.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, run_dir: Path) -> "BenchmarkConfig":
        return cls(**json.loads((run_dir / "config.json").read_text()))


# ── Pipeline ──────────────────────────────────────────────────────────────────

class BenchmarkPipeline:
    """
    Runs the full proposal → worker → eval pipeline.

    Writes all state to disk so the TUI can monitor live.
    Can also be used to resume/reload a previous run.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        task: AbstractBenchmarkTask,
        run_dir: Path,
        on_state_change: Callable[[int, SampleState], None] | None = None,
        slot_pool=None,  # benchmark.slot_pool.SlotPool | None
    ):
        self.config = config
        self.task = task
        self.run_dir = run_dir
        self.on_state_change = on_state_change
        self._stop_event = threading.Event()
        self._samples: list[SampleState] = [
            SampleState(i) for i in range(config.n_samples)
        ]
        self._lock = threading.Lock()
        self._slot_pool = slot_pool  # if set, overrides config.gpu_device/machine per sample

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the pipeline in background threads."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(self.run_dir)
        # Register in the central HQ file so the dashboard can discover this run
        try:
            from benchmark.registry import register_run as _register
            _register(self.run_dir.parent.parent, self.config)
        except Exception:
            pass
        t = threading.Thread(target=self._run_all, daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop_event.set()

    def wait(self, poll_interval: float = 5.0) -> None:
        """Block until every sample reaches DONE or ERROR."""
        while True:
            with self._lock:
                samples = list(self._samples)
            if all(s.step in (PipelineStep.DONE, PipelineStep.ERROR) for s in samples):
                return
            time.sleep(poll_interval)

    @property
    def samples(self) -> list[SampleState]:
        return list(self._samples)

    def get_sample_dir(self, i: int) -> Path:
        return self.run_dir / f"sample_{i:02d}"

    @classmethod
    def load(cls, run_dir: Path, task: AbstractBenchmarkTask,
             on_state_change: Callable | None = None) -> "BenchmarkPipeline":
        """Reload a pipeline from a run directory (for TUI review of past runs)."""
        config = BenchmarkConfig.load(run_dir)
        pipe = cls(config, task, run_dir, on_state_change)
        # Load existing sample states
        for i in range(config.n_samples):
            s = SampleState.load(run_dir / f"sample_{i:02d}")
            if s is not None:
                pipe._samples[i] = s
        return pipe

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update(self, i: int, **kwargs) -> None:
        with self._lock:
            s = self._samples[i]
            for k, v in kwargs.items():
                setattr(s, k, v)
        sample_dir = self.get_sample_dir(i)
        sample_dir.mkdir(parents=True, exist_ok=True)
        s.save(sample_dir)
        if self.on_state_change:
            self.on_state_change(i, self._samples[i])

    def _run_all(self) -> None:
        """Generate proposals and dispatch workers immediately after each generation."""
        threads = []
        for i in range(self.config.n_samples):
            if self._stop_event.is_set():
                break
            if self._slot_pool is not None:
                # Generation also uses the global slot pool so model inference
                # cannot compete with worker evals on the same GPUs.
                caller = f"sample_{i:02d}:generate"
                with self._slot_pool.slot(caller=caller) as slot:
                    record, proposal = self._generate_one(
                        i, gpu_device=slot.gpu, machine=slot.machine
                    )
            else:
                record, proposal = self._generate_one(i)
            if proposal is None:
                continue
            if self._slot_pool is not None:
                # Dynamic slot assignment: each worker acquires a free (machine, gpu) slot
                def _worker_thread(idx=i, rec=record, prop=proposal):
                    caller = f"sample_{idx:02d}"
                    with self._slot_pool.slot(caller=caller) as slot:
                        self._run_worker(idx, rec, prop,
                                         gpu_device=slot.gpu, machine=slot.machine)
            else:
                # Legacy: fixed slot per run, bounded by max_workers semaphore
                sem = getattr(self, "_sem", None)
                if sem is None:
                    self._sem = threading.Semaphore(self.config.max_workers)
                    sem = self._sem

                def _worker_thread(idx=i, rec=record, prop=proposal, _sem=sem):
                    with _sem:
                        self._run_worker(idx, rec, prop)
            t = threading.Thread(target=_worker_thread, daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def _generate_one(
        self,
        i: int,
        gpu_device: str | None = None,
        machine: str | None = None,
    ) -> tuple[dict | None, str | None]:
        """Build prompt and generate a proposal for sample i."""
        sample_dir = self.get_sample_dir(i)
        sample_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        self._update(i, step=PipelineStep.BUILDING_PROMPT)

        # Build prompt record from frontline papers
        record = self._build_prompt_record(i)
        if record is None:
            self._update(i, step=PipelineStep.ERROR, error="failed to build prompt record",
                         elapsed_s=time.time() - t0)
            return None, None

        (sample_dir / "prompt_record.json").write_text(
            json.dumps({k: v for k, v in record.items() if k != "refs_raw"},
                       indent=2, ensure_ascii=False)
        )

        self._update(i, step=PipelineStep.GENERATING)

        # Generate proposal
        proposal = self._generate_proposal(
            record, i, gpu_device=gpu_device, machine=machine
        )
        if proposal is None:
            self._update(i, step=PipelineStep.ERROR, error="proposal generation failed",
                         elapsed_s=time.time() - t0)
            return None, None

        (sample_dir / "proposal.txt").write_text(proposal)
        return record, proposal

    def _build_prompt_record(self, i: int) -> dict | None:
        """
        Return the cached prompt for this run's (task, strategy).

        All samples share the same prompt — the cache is built once and reused.
        The sample index i is unused but kept for API compatibility.
        """
        try:
            from benchmark.prompt_cache import get_or_build
            runs_dir = _REPO / "runs"
            entry = get_or_build(self.config.task_name, self.config.strategy, runs_dir)
            return {
                "system":        entry["system"],
                "prompt":        entry["prompt"],
                "strategy":      self.config.strategy,
                "n_refs":        entry.get("n_refs", 0),
                "frontline_ids": entry.get("frontline_ids", []),
            }
        except Exception:
            import traceback
            traceback.print_exc()
            return None

    def _generate_proposal(
        self,
        record: dict,
        i: int,
        gpu_device: str | None = None,
        machine: str | None = None,
    ) -> str | None:
        """Call the proposal model to generate a proposal."""
        try:
            if self.config.is_api:
                from probe import generate_baseline
                proposal, _ = generate_baseline(record, model=self.config.api_model, max_tokens=4096)
                return proposal
            else:
                # Run generation in a subprocess with the correct CUDA-enabled python
                # (the loongflow_ml env has cu128/CUDA=True; default python may not)
                python_bin = self.config.conda_env or \
                    "/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python"
                return self._generate_via_subprocess(
                    record, python_bin, gpu_device=gpu_device, machine=machine
                )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return None

    def _generate_via_subprocess(
        self,
        record: dict,
        python_bin: str,
        gpu_device: str | None = None,
        machine: str | None = None,
    ) -> str | None:
        """Generate a proposal by calling probe.py in a subprocess with the given python."""
        machine = self.config.machine if machine is None else machine
        machine = machine or ""
        gpu = self.config.gpu_device if gpu_device is None else gpu_device

        # Acquire a per-(machine, gpu) lock so only one generation subprocess runs
        # on each GPU at a time.  Then wait until there is enough free VRAM.
        gen_lock = _get_gen_lock(machine, gpu)
        with gen_lock:
            # Poll until free VRAM is sufficient (handles residual memory from
            # previous subprocess that may not have been freed yet).
            while True:
                free_mb = _free_vram_mb(machine, gpu)
                if free_mb == 0 or free_mb >= _GEN_VRAM_MIN_MB:
                    # free_mb==0 means nvidia-smi failed; proceed anyway
                    break
                print(
                    f"[gen] waiting for VRAM on {machine or 'local'}:GPU{gpu} "
                    f"— free={free_mb}MB < {_GEN_VRAM_MIN_MB}MB, "
                    f"retrying in {_GEN_VRAM_POLL_S}s",
                    flush=True,
                )
                time.sleep(_GEN_VRAM_POLL_S)

            return self._generate_via_subprocess_inner(record, python_bin, machine, gpu)

    def _generate_via_subprocess_inner(
        self, record: dict, python_bin: str, machine: str, gpu: str
    ) -> str | None:
        # Use a directory under run_dir (NFS-accessible) so remote SSH workers can read/write it
        tmp_nfs = self.run_dir / "_gen_tmp"
        tmp_nfs.mkdir(parents=True, exist_ok=True)
        import uuid as _uuid
        token = _uuid.uuid4().hex[:8]
        record_path = tmp_nfs / f"record_{token}.json"
        out_path = tmp_nfs / f"proposal_{token}.json"
        try:
            record_path.write_text(json.dumps({
                k: v for k, v in record.items() if k != "refs_raw"
            }))
            gen_script = str(_REPO / "scripts" / "probe.py")
            local_cmd = [
                python_bin, gen_script,
                "--from-json", str(record_path),
                "--checkpoint", self.config.checkpoint,
                "--temperature", str(self.config.temperature),
                "--save", str(out_path),
                "--quiet",
            ]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}

            if machine and not _is_local(machine):
                env_prefix = f"CUDA_VISIBLE_DEVICES={gpu}"
                for k in _FORWARDED_ENV_VARS:
                    if k in os.environ:
                        v = os.environ[k].replace("'", "'\\''")
                        env_prefix += f" {k}='{v}'"
                escaped = " ".join(f"'{a}'" for a in local_cmd)
                # cd to repo root first so probe.py can find configs/base.yaml
                run_cmd = ["ssh", machine,
                           f"cd '{_REPO}' && {env_prefix} {escaped}"]
                proc = subprocess.run(run_cmd, capture_output=True, text=True, timeout=600)
            else:
                proc = subprocess.run(
                    local_cmd, capture_output=True, text=True, timeout=600, env=env,
                    cwd=str(_REPO),
                )

            if not out_path.exists():
                # Log stderr to _gen_tmp for post-mortem debugging
                if proc.stderr or proc.stdout:
                    err_log = tmp_nfs / f"err_{token}.txt"
                    err_log.write_text((proc.stdout or "") + (proc.stderr or ""))
                return None

            data = json.loads(out_path.read_text())
            return data.get("response")
        except Exception as exc:
            import traceback
            traceback.print_exc()
        finally:
            for p in (record_path, out_path):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        return None

    def _run_worker(self, i: int, record: dict, proposal: str,
                    gpu_device: str | None = None,
                    machine: str | None = None) -> None:
        """Run the Claude Code worker for sample i.

        gpu_device / machine override config values when provided by a SlotPool.
        """
        sample_dir = self.get_sample_dir(i)
        t0 = time.time()
        self._update(i, step=PipelineStep.RUNNING_WORKER)

        # Set up workspace
        workspace = sample_dir / "workspace"
        _gpu = gpu_device if gpu_device is not None else self.config.gpu_device
        _machine = machine if machine is not None else (self.config.machine or "")
        try:
            self.task.setup_workspace(workspace)
        except Exception as exc:
            self._update(i, step=PipelineStep.ERROR,
                         error=f"workspace setup failed: {exc}",
                         elapsed_s=time.time() - t0)
            return

        # Inject CUDA_VISIBLE_DEVICES into run.sh so the training script uses the
        # assigned slot GPU.  Claude Code's shell-snapshot mechanism resets env vars,
        # so we cannot rely on the inherited environment alone.
        run_sh = workspace / "run.sh"
        if run_sh.exists():
            content = run_sh.read_text()
            if "export CUDA_VISIBLE_DEVICES=" not in content:
                content = content.replace(
                    "#!/bin/bash\nset -e\n",
                    f"#!/bin/bash\nset -e\nexport CUDA_VISIBLE_DEVICES={_gpu}\n",
                )
                run_sh.write_text(content)

        # Build worker prompt
        task_context = self.task.task_context()
        worker_prompt = _WORKER_PROMPT_TEMPLATE.format(
            PROPOSAL_TEXT=proposal,
            TASK_CONTEXT=task_context,
        )
        (sample_dir / "worker_prompt.txt").write_text(worker_prompt)

        # Run Claude Code CLI
        local_claude_cmd = [
            self.config.claude_cmd,
            "-p",
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", "30",
            "--allowedTools", "Bash,Read,Edit,Write",
        ]

        log_file = sample_dir / "worker.log"
        prompt_file = sample_dir / "worker_prompt.txt"
        result_file = workspace / "result.json"

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": _gpu,
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}

        if _machine and not _is_local(_machine):
            # Remote: SSH to machine; workspace and prompt are on shared NFS so paths match
            exports = [
                f"export CUDA_VISIBLE_DEVICES={_gpu}",
                "export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
            ]
            for k in _FORWARDED_ENV_VARS:
                if k in os.environ:
                    v = os.environ[k].replace("'", "'\\''")
                    exports.append(f"export {k}='{v}'")
            remote_shell = (
                f"{'; '.join(exports)}; cd '{workspace}' && {self.config.claude_cmd} -p"
                f" --output-format stream-json --verbose --max-turns 30"
                f" --allowedTools Bash,Read,Edit,Write"
                f" < '{prompt_file}'"
            )
            with open(log_file, "w") as log_f:
                proc = subprocess.Popen(
                    ["ssh", _machine, remote_shell],
                    stdout=log_f, stderr=subprocess.STDOUT,
                )
        else:
            with open(prompt_file) as stdin_f, open(log_file, "w") as log_f:
                proc = subprocess.Popen(
                    local_claude_cmd, stdin=stdin_f, stdout=log_f, stderr=subprocess.STDOUT,
                    cwd=str(workspace), env=env,
                )

        try:
            proc.wait(timeout=WORKER_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            self._update(i, step=PipelineStep.ERROR,
                         error=f"worker timeout after {WORKER_TIMEOUT}s",
                         elapsed_s=time.time() - t0)
            return
        except BaseException:
            # Thread interrupted (KeyboardInterrupt, SystemExit, or sweep cancellation):
            # terminate the worker subprocess so it doesn't become an orphan.
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise

        elapsed = time.time() - t0

        # Copy result.json up to sample_dir for easy access
        if result_file.exists():
            shutil.copy2(result_file, sample_dir / "result.json")
            # Also copy eval.log if it was written to workspace
            eval_log_src = workspace / "eval.log"
            if eval_log_src.exists():
                shutil.copy2(eval_log_src, sample_dir / "eval.log")
            # Copy editable_region.py to sample_dir
            er = workspace / "editable_region.py"
            if er.exists():
                shutil.copy2(er, sample_dir / "editable_region.py")

        # Parse result
        if not (sample_dir / "result.json").exists():
            self._update(i, step=PipelineStep.ERROR,
                         error="result.json not written",
                         elapsed_s=elapsed)
            return

        try:
            result = json.loads((sample_dir / "result.json").read_text())
        except Exception:
            self._update(i, step=PipelineStep.ERROR,
                         error="result.json invalid JSON",
                         elapsed_s=elapsed)
            return

        val_metric = result.get("val_metric")
        error = result.get("error")

        # Verify HMAC
        if val_metric is not None and "_sig" in result:
            import hashlib as _hl
            import hmac as _hmac
            secret = os.environ.get("BENCHMARK_HMAC_KEY", "benchmark-eval-secret")
            payload = f"{float(val_metric):.6f}"
            expected = _hmac.new(secret.encode(), payload.encode(), _hl.sha256).hexdigest()
            if not _hmac.compare_digest(expected, result["_sig"]):
                val_metric = None
                error = "HMAC mismatch — result may be fabricated"
        elif val_metric is not None and "_sig" not in result:
            val_metric = None
            error = "result.json lacks HMAC signature"

        if val_metric is not None:
            improvement = self.task.improvement(float(val_metric))
            passed = self.task.passed(float(val_metric))
            self._update(i, step=PipelineStep.DONE,
                         improvement=improvement, passed=passed,
                         elapsed_s=elapsed)
        else:
            self._update(i, step=PipelineStep.ERROR,
                         error=error or "val_metric is null",
                         elapsed_s=elapsed)


# ── Summary helpers ───────────────────────────────────────────────────────────

def compute_summary(samples: list[SampleState], task: AbstractBenchmarkTask) -> dict:
    """Compute pass@k and mean delta from completed samples."""
    done = [s for s in samples if s.step == PipelineStep.DONE and s.improvement is not None]
    errors = [s for s in samples if s.step == PipelineStep.ERROR]
    total = len(samples)

    if not done:
        return {"n_total": total, "n_done": 0, "n_errors": len(errors),
                "pass_at_1": 0.0, "mean_improvement": None}

    passed = [s for s in done if s.passed]
    improvements = [s.improvement for s in done]

    # pass@k via bootstrap (simplified: exact for k=1)
    n_pass = len(passed)
    pass_at_1 = n_pass / len(done) if done else 0.0

    def pass_at_k(k: int) -> float:
        if len(done) < k:
            return float("nan")
        # P(at least 1 of k passes) = 1 - C(n_fail, k)/C(n_done, k)
        import math
        n = len(done)
        n_fail = len(done) - n_pass
        if n_fail < k:
            return 1.0
        return 1.0 - math.comb(n_fail, k) / math.comb(n, k)

    return {
        "n_total": total,
        "n_done": len(done),
        "n_errors": len(errors),
        "n_passed": n_pass,
        "pass_at_1": pass_at_1,
        "pass_at_3": pass_at_k(3),
        "pass_at_5": pass_at_k(5),
        "mean_improvement": sum(improvements) / len(improvements),
        "baseline_metric": task.baseline_metric(),
        "pass_threshold": task.pass_threshold,
    }
