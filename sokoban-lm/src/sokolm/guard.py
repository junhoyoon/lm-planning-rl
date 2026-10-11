"""Resource guard for sokolm experiments on a WSL2 workstation (a Windows host whose one GPU, an RTX 5090, also drives
the monitor, and a WSL VM with capped memory).

The CLI starts it as a separate process for every experiment command (`python -m sokolm.guard --pid <cli pid>`).

Before the run it checks that Windows and WSL have free RAM and that the GPU is idle, waiting up to START_WAIT_S
(a previous run may still be releasing its memory), and refuses the run otherwise. During the run it samples, every
INTERVAL_S:
  Windows: available RAM, committed share, pagefile use, the WSL VM's working set (one long-lived PowerShell);
  WSL: MemAvailable, swap in use, the run's RSS (/proc);
  GPU: memory in use on the whole card (Windows apps included), utilization, temperature, power, the SM and memory
  clocks and the active clock-event reasons (nvidia-smi).
Every sample is a line of RUNS/resources/<start>-<command>.csv, written at once and fsynced, so the log
survives a forced power-off. When a limit below is crossed, or Windows or the GPU has not answered for STALE_S, the
run is stopped (SIGTERM, SIGKILL after KILL_WAIT_S) and the reason goes to the log and to stderr.

Why: during one run the screen froze and the machine had to be switched off; Windows had held the WSL disk's I/O
for up to 36 s, and nothing had recorded RAM, so the cause could not be established.

`python -m sokolm.guard --check` (or `python -m sokolm resources`) prints the current numbers for the estimate
before a run.

Only for WSL2: without PowerShell the Windows sampler cannot start and every run is refused. On other machines set
SOKOLM_GUARD=0 and the CLI runs experiments without the guard.
"""

import argparse
import base64
import datetime
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from sokolm.paths import RUNS

INTERVAL_S = 2.0
STALE_S = 20.0
KILL_WAIT_S = 10.0
START_WAIT_S = 30.0

# before the run
START_MIN_WINDOWS_AVAILABLE_MB = 16 * 1024
START_MIN_WSL_AVAILABLE_MB = 8 * 1024
START_MAX_GPU_USED_MIB = 4 * 1024  # the Windows desktop and apps alone use about 2 GiB
# during the run
MIN_WINDOWS_AVAILABLE_MB = 8 * 1024
MIN_WSL_AVAILABLE_MB = 2 * 1024
MAX_GPU_USED_MIB = 26 * 1024  # of 32 GiB; an experiment caps its own process at 16 GiB
MAX_RSS_MB = float(os.environ.get("SOKOLM_MAX_RSS_GB", "12")) * 1024

POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"
# Clocks are logged because the steps of one run were ~8% slower than those of a run like it, with nothing
# recorded to explain it.
# gpu_clock_events is the bitmask of clocks_event_reasons.active (nvml.h), e.g. 0x4 SW power cap, 0x20/0x40 thermal
# slowdown, 0x400 reliability (active on this card even when idle).
GPU_FIELDS = ("gpu_used_mib", "gpu_total_mib", "gpu_util_pct", "gpu_temp_c", "gpu_power_w", "gpu_sm_mhz",
              "gpu_mem_mhz", "gpu_clock_events")
GPU_QUERY = ("memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,"
             "clocks_event_reasons.active")
COLUMNS = ("time", "elapsed_s", "win_available_mb", "win_committed_pct", "win_pagefile_pct", "wsl_vm_mb",
           "win_sample_age_s", "wsl_available_mb", "wsl_swap_used_mb", "rss_mb", *GPU_FIELDS, "gpu_query_ms",
           "gpu_sample_age_s", "stop")

# Performance counters are created once and read every interval (cheap). The loop ends when the guard closes the
# pipe (the write fails) or after __HOURS__ hours, so no PowerShell is left behind on the Windows side.
_WINDOWS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$inv = [Globalization.CultureInfo]::InvariantCulture
$avail = New-Object Diagnostics.PerformanceCounter('Memory', 'Available MBytes')
$commit = New-Object Diagnostics.PerformanceCounter('Memory', '% Committed Bytes In Use')
$page = New-Object Diagnostics.PerformanceCounter('Paging File', '% Usage', '_Total')
$end = (Get-Date).AddHours(__HOURS__)
while ((Get-Date) -lt $end) {
    $vm = (Get-Process -Name vmmemWSL -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum
    [Console]::Out.WriteLine([string]::Format($inv, '{0:F0},{1:F1},{2:F1},{3:F0}',
        $avail.NextValue(), $commit.NextValue(), $page.NextValue(), $vm / 1MB))
    [Console]::Out.Flush()
    Start-Sleep -Milliseconds __MS__
}
"""


def parse_windows(line: str) -> dict:
    avail, commit, page, vm = (float(v) for v in line.strip().split(","))
    return {"win_available_mb": avail, "win_committed_pct": commit, "win_pagefile_pct": page, "wsl_vm_mb": vm}


def parse_meminfo(text: str) -> dict:
    kb = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in text.splitlines() if ":" in line)}
    return {"wsl_available_mb": kb["MemAvailable"] / 1024, "wsl_swap_used_mb": (kb["SwapTotal"] - kb["SwapFree"]) / 1024}


def parse_gpu(line: str) -> dict:
    """One line of nvidia-smi --format=csv,noheader,nounits; fields it cannot report ([N/A]) become None. The
    clock-event bitmask comes as hex (0x0000000000000400) and becomes an int."""
    def num(field: str, v: str):
        try:
            return int(v, 16) if field == "gpu_clock_events" else float(v)
        except ValueError:
            return None
    return {k: num(k, v.strip()) for k, v in zip(GPU_FIELDS, line.strip().split(","))}


def rss_mb(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return None


class _Sampler:
    """A thread that keeps the latest sample of a source and when it arrived."""

    def __init__(self):
        self.latest, self.at, self.error = None, None, ""

    def put(self, sample: dict):
        self.latest, self.at = sample, time.monotonic()

    def age(self) -> float | None:
        return None if self.at is None else time.monotonic() - self.at


class WindowsSampler(_Sampler):
    def __init__(self, hours: float = 24):
        super().__init__()
        script = _WINDOWS_SCRIPT.replace("__HOURS__", str(hours)).replace("__MS__", str(int(INTERVAL_S * 1000)))
        self.proc = subprocess.Popen(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand",
             base64.b64encode(script.encode("utf-16-le")).decode()],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_errors, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.put(parse_windows(line))
            except ValueError:
                self.error = line.strip()[:300]

    def _read_errors(self):
        for line in self.proc.stderr:
            if line.strip() and not line.lstrip().startswith(("#< CLIXML", "<Objs")):  # PowerShell progress records
                self.error = (self.error + " " + line.strip())[-300:]

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class GpuSampler(_Sampler):
    def __init__(self):
        super().__init__()
        self.last_query_ms = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            t = time.monotonic()
            try:
                out = subprocess.run([NVIDIA_SMI, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=STALE_S, check=True).stdout
                self.last_query_ms = round((time.monotonic() - t) * 1000)
                self.put({**parse_gpu(out.splitlines()[0]), "gpu_query_ms": self.last_query_ms})
            except (subprocess.SubprocessError, OSError, IndexError) as e:
                self.error = f"nvidia-smi: {type(e).__name__}"
            time.sleep(max(0.0, INTERVAL_S - (time.monotonic() - t)))


def sample(win: WindowsSampler, gpu: GpuSampler, pid: int | None) -> dict:
    s = {"time": datetime.datetime.now().strftime("%H:%M:%S")}
    s.update(win.latest or {})
    s["win_sample_age_s"] = win.age()
    s.update(parse_meminfo(Path("/proc/meminfo").read_text()))
    s["rss_mb"] = rss_mb(pid) if pid else None
    s.update(gpu.latest or {})
    s["gpu_sample_age_s"] = gpu.age()
    return s


def problems(s: dict, start: bool) -> list[str]:
    """The limits a sample crosses (empty if none); `start` applies the stricter limits for starting a run."""
    win_min, wsl_min, gpu_max = ((START_MIN_WINDOWS_AVAILABLE_MB, START_MIN_WSL_AVAILABLE_MB, START_MAX_GPU_USED_MIB)
                                 if start else (MIN_WINDOWS_AVAILABLE_MB, MIN_WSL_AVAILABLE_MB, MAX_GPU_USED_MIB))
    out = []
    if s.get("win_available_mb") is None:
        out.append("Windows RAM could not be read")
    elif s["win_available_mb"] < win_min:
        out.append(f"Windows available RAM {s['win_available_mb']:.0f} MB < {win_min} MB")
    if (s.get("win_sample_age_s") or 0) > STALE_S:
        out.append(f"no Windows sample for {s['win_sample_age_s']:.0f} s")
    if s["wsl_available_mb"] < wsl_min:
        out.append(f"WSL available RAM {s['wsl_available_mb']:.0f} MB < {wsl_min} MB")
    if s.get("gpu_used_mib") is None:
        out.append("GPU memory could not be read")
    elif s["gpu_used_mib"] > gpu_max:
        out.append(f"GPU memory in use {s['gpu_used_mib']:.0f} MiB > {gpu_max} MiB")
    if (s.get("gpu_sample_age_s") or 0) > STALE_S:
        out.append(f"nvidia-smi has not answered for {s['gpu_sample_age_s']:.0f} s")
    if not start and (s.get("rss_mb") or 0) > MAX_RSS_MB:
        out.append(f"run RSS {s['rss_mb']:.0f} MB > {MAX_RSS_MB:.0f} MB")
    return out


class Log:
    """CSV lines handed to the kernel at once; a thread fsyncs every INTERVAL_S, so a stalled disk delays only that."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", buffering=1)
        self.f.write(",".join(COLUMNS) + "\n")
        threading.Thread(target=self._sync, daemon=True).start()

    def row(self, s: dict):
        def fmt(v):
            return "" if v is None else f"{v:.1f}" if isinstance(v, float) else str(v).replace(",", ";")
        self.f.write(",".join(fmt(s.get(c)) for c in COLUMNS) + "\n")

    def note(self, text: str):
        self.f.write(f"# {text}\n")

    def _sync(self):
        while True:
            time.sleep(INTERVAL_S)
            try:
                os.fsync(self.f.fileno())
            except (OSError, ValueError):
                return

    def close(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()


def log_path(command: str) -> Path:
    return RUNS / "resources" / f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{command}.csv"


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


def _stop(pid: int) -> str:
    os.kill(pid, signal.SIGTERM)
    t = time.monotonic() + KILL_WAIT_S
    while time.monotonic() < t:
        if not _alive(pid):
            return "SIGTERM"
        time.sleep(0.2)
    os.kill(pid, signal.SIGKILL)
    return "SIGKILL"


def _start_check(win: WindowsSampler, gpu: GpuSampler) -> tuple[dict, list[str]]:
    """Sample until the start limits hold or START_WAIT_S has passed."""
    deadline = time.monotonic() + START_WAIT_S
    while True:
        s = sample(win, gpu, None)
        bad = problems(s, start=True)
        if not bad or time.monotonic() > deadline or win.proc.poll() is not None:
            return s, bad
        time.sleep(0.5 if s.get("win_available_mb") is None or s.get("gpu_used_mib") is None else INTERVAL_S)


def guard(pid: int, path: Path) -> int:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C reaches the whole process group; the CLI stops the guard
    finish = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: finish.set())
    win, gpu, log = WindowsSampler(), GpuSampler(), Log(path)
    t0 = time.monotonic()
    s, bad = _start_check(win, gpu)
    s["elapsed_s"] = 0.0
    if bad:
        reason = "; ".join(bad + [f"({e})" for e in (win.error, gpu.error) if e])
        s["stop"] = f"refused: {reason}"
        log.row(s)
        log.close()
        win.close()
        print(f"refuse {reason}", flush=True)
        return 1
    log.row(s)
    print("ready", flush=True)
    rows, stopped = [s], None
    while not finish.wait(INTERVAL_S) and _alive(pid):
        s = sample(win, gpu, pid)
        s["elapsed_s"] = time.monotonic() - t0
        bad = problems(s, start=False)
        if bad:
            stopped = "; ".join(bad)
            s["stop"] = stopped
            log.row(s)
            print(f"[guard] stopping the run (pid {pid}): {stopped}", file=sys.stderr, flush=True)
            log.note(f"stopped the run with {_stop(pid)}: {stopped}")
            rows.append(s)
            break
        log.row(s)
        rows.append(s)

    def extreme(key, f):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return f"{f(vals):.0f}" if vals else "?"
    summary = (f"min Windows available {extreme('win_available_mb', min)} MB, max WSL VM {extreme('wsl_vm_mb', max)} "
               f"MB, max run RSS {extreme('rss_mb', max)} MB, max GPU in use {extreme('gpu_used_mib', max)} MiB, "
               f"max GPU {extreme('gpu_temp_c', max)} C, {len(rows)} samples"
               + (f", stopped: {stopped}" if stopped else ""))
    log.note(summary)
    log.close()
    win.close()
    print(f"[guard] {summary}; log {path}", file=sys.stderr, flush=True)
    return 0


def check() -> int:
    """Print the current numbers and whether a run would be allowed to start."""
    win, gpu = WindowsSampler(hours=0.05), GpuSampler()
    try:
        s, bad = _start_check(win, gpu)
    finally:
        win.close()

    def v(key, digits=0):
        return "?" if s.get(key) is None else f"{s[key]:.{digits}f}"
    print(f"Windows: available {v('win_available_mb')} MB, committed {v('win_committed_pct', 1)} %, "
          f"pagefile {v('win_pagefile_pct', 1)} %, WSL VM working set {v('wsl_vm_mb')} MB")
    print(f"WSL: available {v('wsl_available_mb')} MB, swap in use {v('wsl_swap_used_mb')} MB")
    events = "?" if s.get("gpu_clock_events") is None else hex(s["gpu_clock_events"])
    print(f"GPU: in use {v('gpu_used_mib')} / {v('gpu_total_mib')} MiB, utilization {v('gpu_util_pct')} %, "
          f"{v('gpu_temp_c')} C, {v('gpu_power_w', 1)} W, SM clock {v('gpu_sm_mhz')} MHz, memory clock "
          f"{v('gpu_mem_mhz')} MHz, clock events {events}")
    print("start check: " + ("ok" if not bad else "would refuse: " + "; ".join(bad)))
    return 0 if not bad else 1


def main():
    p = argparse.ArgumentParser(prog="sokolm.guard")
    p.add_argument("--pid", type=int, help="the experiment process to watch")
    p.add_argument("--log", type=Path, help="CSV log path")
    p.add_argument("--check", action="store_true", help="print the current numbers and the start check, then exit")
    a = p.parse_args()
    if a.check:
        sys.exit(check())
    if a.pid is None or a.log is None:
        p.error("--pid and --log are required unless --check")
    sys.exit(guard(a.pid, a.log))


if __name__ == "__main__":
    main()
