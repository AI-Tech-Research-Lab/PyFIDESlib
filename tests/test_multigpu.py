#!/usr/bin/env python3
"""Multi-GPU regression suite.

Covers the failures found in the 2026-09 multi-GPU investigation (docs/multigpu-plan.md):

  F3  more devices than special primes (#GPUs > K) aborted mid-computation with
      std::out_of_range out of an OpenMP region
  F5  a device list that is not 0,1,2,... crashed at context teardown with
      'invalid resource handle' -- the GPU memory pool was indexed by the partition
      index where it should have been the device ordinal
  F4  the extended hoisted rotation (AccumulateSum/Broadcast) returning NaN under
      contention; --contend adds a synthetic load to hunt for what is left of it

Every case runs in its own subprocess via mgpu_case.py, because the failures being
guarded against kill the process rather than raise: only a subprocess boundary
distinguishes "wrong value" from "died", and reports the second as a real failure.

    python tests/test_multigpu.py                   # everything the machine can run
    python tests/test_multigpu.py --devices 0,2,3   # restrict to these physical GPUs
    python tests/test_multigpu.py --group k-table   # one group
    python tests/test_multigpu.py --contend         # add the contention sweep (slow)

--devices is an allow-list, exported as CUDA_VISIBLE_DEVICES for every case, so no case
ever touches a GPU outside it. Within that list the cases vary what they hand to
SetDevices: subsets and permutations of 0..n-1, not just prefixes. That distinction is
the whole point for F5 -- CUDA_VISIBLE_DEVICES renumbers, so a prefix always looks like
the identity mapping to the library, which is exactly the case where the pool-index bug
does not fire. It takes a gap (SetDevices([0,2]) of three visible GPUs) to expose it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = Path(__file__).resolve().parent / "mgpu_case.py"


def visible_gpus() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in out.split() if line.strip().isdigit()]


class Case:
    """One (SetDevices argument, parameters, operation) point.

    `visible` is the allow-list exported as CUDA_VISIBLE_DEVICES and is the same for
    every case in a run; `ids` are the indices into it that the case passes to
    SetDevices, and they need not be a prefix -- see the module docstring.
    """

    def __init__(self, group, visible, ids, cfg, expect_warning=None):
        self.group = group
        self.visible = list(visible)
        self.ids = list(ids)
        self.cfg = dict(cfg, devices=list(ids))
        self.expect_warning = expect_warning
        self.status = "pending"
        self.detail = ""
        self.seconds = 0.0

    @property
    def phys(self):
        return [self.visible[i] for i in self.ids]

    def __str__(self):
        c = self.cfg
        return (f"{self.group}: SetDevices({self.ids})=gpu{self.phys} dnum={c['dnum']} "
                f"depth={c['depth']} logN={c.get('logN', 14)} op={c.get('op', 'accumulate')}")

    def run(self, timeout, load_pids=()):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in self.visible)
        start = time.monotonic()
        try:
            p = subprocess.run([sys.executable, str(CASE), json.dumps(self.cfg)],
                               capture_output=True, text=True, timeout=timeout,
                               env=env, cwd=str(ROOT))
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            self.status, self.detail = "FAIL", f"timed out after {timeout}s"
            self.seconds = time.monotonic() - start
            return
        self.seconds = time.monotonic() - start

        checks = [json.loads(l) for l in out.splitlines() if l.startswith('{"check"')]
        bad = [c for c in checks if not c["ok"]]

        # A negative return code is a signal: the abort (F3) and the teardown crash (F5)
        # both arrive that way, and neither prints a failing check.
        if rc < 0:
            self.status, self.detail = "FAIL", f"killed by signal {-rc}"
        elif bad:
            self.status = "FAIL"
            self.detail = "; ".join(f"{c['check']}: got {c.get('got')!r}, "
                                    f"want {c.get('expected')!r}" for c in bad)
        elif not any(c["check"] == "teardown" for c in checks):
            self.status, self.detail = "FAIL", f"no teardown check (rc={rc})"
        elif "Cuda failure" in err:
            line = next(l for l in err.splitlines() if "Cuda failure" in l)
            self.status, self.detail = "FAIL", line.strip()
        else:
            self.status = "ok"

        # The zero-special-prime diagnostic is part of the contract, not decoration:
        # a device that owns none takes no part in key switching and the user has to
        # be told. Check it fires exactly where it should.
        if self.expect_warning is not None and self.status == "ok":
            warned = "take no part in key switching" in err
            if warned != self.expect_warning:
                self.status = "FAIL"
                self.detail = ("expected the zero-special-prime diagnostic, none printed"
                               if self.expect_warning else
                               "unexpected zero-special-prime diagnostic")


def build_cases(gpus, contend):
    """gpus is the allow-list; a case's device argument indexes into it (0..n-1)."""
    n = len(gpus)
    ids = list(range(n))
    cases = []

    # --- k-table: #GPUs against K = ceil((L+1)/dnum). Everything above the K threshold
    # used to abort inside LimbPartition::add; now it must compute and warn instead.
    for (depth, dnum), k in itertools.product(
            [(4, 5), (4, 3), (8, 3), (12, 5), (12, 3)], range(1, n + 1)):
        K = -(-(depth + 1) // dnum)
        cases.append(Case("k-table", gpus, ids[:k],
                          {"dnum": dnum, "depth": depth, "logN": 14, "sizes": [2, 4, 8]},
                          expect_warning=k > K))

    # --- device-sets: every subset of the allow-list, plus its reverse. Prefixes alone
    # would miss the pool-index bug entirely, which needs a device list that is not
    # 0,1,...,n-1 -- a gap or a permutation.
    subsets = []
    for size in range(1, n + 1):
        for combo in itertools.combinations(ids, size):
            subsets.append(list(combo))
            if size > 1:
                subsets.append(list(reversed(combo)))
    for sub in subsets:
        cases.append(Case("device-sets", gpus, sub,
                          {"dnum": 3, "depth": 24, "logN": 14}))

    # --- ops: the extended hoisted rotation is the fragile path, but a regression in
    # the plain key switch or in a mult chain would otherwise be misread as one of its
    # failures. "mixed" is the workload-shaped one -- every arithmetic op the API offers,
    # interleaved, each intermediate checked -- and is what says whether ordinary use is
    # sound; the other three isolate one path each so a failure is attributable.
    for op in ("mixed", "accumulate", "rotate", "chain"):
        for k in range(1, n + 1):
            cases.append(Case("ops", gpus, ids[:k],
                              {"dnum": 3, "depth": 12, "logN": 14, "op": op}))

    # --- contention: F4 was only ever seen with a saturated peer GPU. Repeat both the
    # isolated extended path and the full mixed workload several times, so an intermittent
    # failure has a chance to appear and so it is visible which operation it lands on.
    if contend:
        for k in range(2, n + 1):
            for _ in range(6):
                cases.append(Case("contention", gpus, ids[:k],
                                  {"dnum": 3, "depth": 24, "logN": 14}))
            for _ in range(4):
                cases.append(Case("contention", gpus, ids[:k],
                                  {"dnum": 3, "depth": 24, "logN": 14,
                                   "op": "mixed", "rounds": 3}))

    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", help="comma-separated physical GPUs to use "
                                      "(default: every GPU nvidia-smi reports)")
    ap.add_argument("--group", action="append",
                    help="run only this group (repeatable): k-table, device-sets, ops, "
                         "contention")
    ap.add_argument("--contend", action="store_true",
                    help="add the contention sweep, and load the GPUs while it runs")
    ap.add_argument("--timeout", type=int, default=1800, help="per-case seconds")
    args = ap.parse_args()

    if args.devices:
        gpus = [int(d) for d in args.devices.split(",")]
    else:
        gpus = visible_gpus()
    if not gpus:
        print("No GPUs found (nvidia-smi unavailable?); nothing to run.")
        return 0
    print(f"Devices: {gpus}")

    cases = build_cases(gpus, args.contend)
    if args.group:
        cases = [c for c in cases if c.group in args.group]
    if not cases:
        print("No cases selected.")
        return 0

    load = None
    if args.contend and any(c.group == "contention" for c in cases):
        load = start_load(gpus)

    try:
        width = max(len(str(c)) for c in cases)
        for i, c in enumerate(cases, 1):
            print(f"[{i:3d}/{len(cases)}] {str(c):<{width}} ", end="", flush=True)
            c.run(args.timeout)
            print(f"{c.status:>4}  {c.seconds:6.1f}s"
                  + (f"  {c.detail}" if c.detail else ""), flush=True)
    finally:
        if load is not None:
            load.terminate()
            load.wait(timeout=30)

    failed = [c for c in cases if c.status != "ok"]
    print(f"\n{len(cases) - len(failed)}/{len(cases)} passed")
    for c in failed:
        print(f"  FAIL {c}\n       {c.detail}")
    return 1 if failed else 0


def start_load(gpus):
    """Saturate the GPUs with an unrelated process, via tests/tools/gpu_spin.

    F4 was reproduced only against a genuinely busy peer. The leading suspects -- the
    CPU-side spin barriers in modupMGPU and the cross-device polling kernels in
    PeerUtils.cu -- fail when the notifying side cannot get SMs, so the stressor is one
    resident block per SM rather than a compute loop that yields between launches.
    Compile it once with:

        nvcc -O2 -arch=sm_90 -o tests/tools/gpu_spin tests/tools/gpu_spin.cu
    """
    binary = ROOT / "tests" / "tools" / "gpu_spin"
    if not binary.exists():
        print(f"  ({binary} not built: running the contention group without extra load)")
        return None
    print(f"  starting spin load on {gpus}")
    p = subprocess.Popen([str(binary), *[str(g) for g in gpus]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)  # let the kernels become resident before the first case starts
    return p


if __name__ == "__main__":
    sys.exit(main())
