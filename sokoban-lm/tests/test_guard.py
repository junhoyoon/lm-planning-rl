"""sokolm.guard: the parsers of the three sources and the limits (no PowerShell, nvidia-smi or GPU needed)."""

from sokolm import guard


def _ok_sample(**changes):
    s = {"win_available_mb": 40000.0, "win_sample_age_s": 1.0, "wsl_available_mb": 20000.0, "gpu_used_mib": 2000.0,
         "gpu_sample_age_s": 1.0, "rss_mb": 3000.0}
    s.update(changes)
    return s


def test_parse_windows():
    assert guard.parse_windows("41234,23.4,0.0,4790\r\n") == {
        "win_available_mb": 41234.0, "win_committed_pct": 23.4, "win_pagefile_pct": 0.0, "wsl_vm_mb": 4790.0}


def test_parse_meminfo():
    text = "MemTotal:       24000000 kB\nMemAvailable:   20480000 kB\nSwapTotal:      1048576 kB\nSwapFree:  524288 kB\n"
    assert guard.parse_meminfo(text) == {"wsl_available_mb": 20000.0, "wsl_swap_used_mb": 512.0}


def test_parse_gpu_with_missing_field():
    assert guard.parse_gpu("1962, 32607, 2, 40, [N/A], 2467, 14001, 0x0000000000000400") == {
        "gpu_used_mib": 1962.0, "gpu_total_mib": 32607.0, "gpu_util_pct": 2.0, "gpu_temp_c": 40.0, "gpu_power_w": None,
        "gpu_sm_mhz": 2467.0, "gpu_mem_mhz": 14001.0, "gpu_clock_events": 0x400}
    assert guard.parse_gpu("1962, 32607, 2, 40, 68.9, [N/A], [N/A], [N/A]")["gpu_clock_events"] is None


def test_limits():
    assert guard.problems(_ok_sample(), start=True) == []
    assert guard.problems(_ok_sample(), start=False) == []
    # 10 GB available: too little to start, enough to keep running
    low = _ok_sample(win_available_mb=10000.0)
    assert len(guard.problems(low, start=True)) == 1 and guard.problems(low, start=False) == []
    # a previous run still holding 6 GiB: refuse to start
    assert guard.problems(_ok_sample(gpu_used_mib=6000.0), start=True)
    for bad in ({"win_available_mb": None}, {"win_sample_age_s": 25.0}, {"wsl_available_mb": 1000.0},
                {"gpu_used_mib": 27000.0}, {"gpu_used_mib": None}, {"gpu_sample_age_s": 25.0},
                {"rss_mb": guard.MAX_RSS_MB + 1}):
        assert len(guard.problems(_ok_sample(**bad), start=False)) == 1, bad
