#!/usr/bin/env python3
"""Regression for orphaned key-switch pointer tables (training run 2101).

Run: python tests/test_bootstrap_memory.py
     CUDA_VISIBLE_DEVICES=0,1 python tests/test_bootstrap_memory.py --devices 0,1

Uses small, non-secure parameters and the locally built extension. CUDA's used
allocation bytes must stay constant after warm-up; FIDESlib's slab counters alone
miss this leak. RSS is reported separately. Each batch also verifies decryption.
The parent checks both a completion marker and the worker's exit status because
native CUDA teardown can otherwise hide a Python failure or crash after success.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MARKER = "BOOTSTRAP_MEMORY_PASS"


def run(args):
    sys.path.insert(0, str(ROOT))
    import fideslib_py as fhe
    import fideslib_py._core as core

    # Resolve the runtime linked to this extension, rather than assuming a CUDA
    # major version or installation path. No GPU allocation interposition.
    cuda = ctypes.CDLL(core.__file__)
    cuda.cudaSetDevice.argtypes = [ctypes.c_int]
    cuda.cudaDeviceGetDefaultMemPool.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    cuda.cudaMemPoolGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

    def check(code):
        if code != 0:
            raise RuntimeError(f"CUDA runtime error {code}")

    devices = [int(d) for d in args.devices.split(",")]
    slots, depth = 4096, 30
    p = fhe.CCParams()
    p.SetSecurityLevel(fhe.HEStd_NotSet)
    p.SetRingDim(8192)
    p.SetMultiplicativeDepth(depth)
    p.SetScalingModSize(59)
    p.SetFirstModSize(60)
    p.SetNumLargeDigits(3)
    p.SetBatchSize(slots)
    p.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    p.SetKeySwitchTechnique(fhe.HYBRID)
    p.SetSecretKeyDist(fhe.SPARSE_ENCAPSULATED)
    p.SetDevices(devices)
    cc = fhe.GenCryptoContext(p)
    for feature in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(feature)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapSetup([4, 4], [0, 0], slots)
    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    cc.LoadContext(keys.publicKey)
    values = [0.01 + 0.0001 * (i % 17) for i in range(slots)]
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(values, 1, depth - 1, slots))

    def batch(count):
        for i in range(count):
            result = cc.EvalBootstrap(ct, 2, 8)
            if i == count - 1:
                pt = cc.Decrypt(keys.secretKey, result)
                pt.SetLength(slots)
                got = pt.GetRealPackedValue()
                if len(got) != slots or not all(math.isfinite(x) for x in got):
                    raise AssertionError("invalid bootstrap output")
                error = max(abs(a - b) for a, b in zip(got, values))
                if error >= 1e-5:
                    raise AssertionError(f"bootstrap error {error}")
                print(json.dumps({"max_error": error}), flush=True)
            del result

    def snapshot(completed):
        cc.Synchronize()
        cc.ClearAuxiliaryPolyPool()
        gc.collect()
        cc.TrimGPUMemoryPool()
        libc = ctypes.CDLL(None)
        trim = getattr(libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
        used = []
        for device in devices:
            check(cuda.cudaSetDevice(device))
            pool = ctypes.c_void_p()
            check(cuda.cudaDeviceGetDefaultMemPool(ctypes.byref(pool), device))
            nbytes = ctypes.c_uint64()
            # cudaMemPoolAttrUsedMemCurrent = 7, stable CUDA runtime enum.
            check(cuda.cudaMemPoolGetAttribute(pool, 7, ctypes.byref(nbytes)))
            used.append(nbytes.value)
        rss_kib = None
        if Path("/proc/self/status").exists():
            with open("/proc/self/status") as handle:
                rss_kib = next(int(line.split()[1]) for line in handle if line.startswith("VmRSS:"))
        print(json.dumps({"completed": completed, "cuda_used_bytes": used,
                          "rss_kib": rss_kib, "objects": cc.GetDeviceObjectCounts()}), flush=True)
        return used

    batch(args.warmup)
    baseline = snapshot(args.warmup)
    for i in range(args.batches):
        batch(args.iterations)
        current = snapshot(args.warmup + (i + 1) * args.iterations)
        if current != baseline:
            raise AssertionError(f"CUDA allocations grew: baseline={baseline}, current={current}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0", help="CUDA ordinals within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.batches) < 1:
        parser.error("warmup, iterations and batches must be positive")
    if args.worker:
        run(args)
        gc.collect()
        print(MARKER, flush=True)
        return 0
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", args.devices)
    # When setting visibility ourselves, remap requested physical devices to
    # ordinal indices. With an existing visibility list, --devices is ordinal.
    argv = sys.argv[1:]
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        argv = [*argv, "--devices", ",".join(str(i) for i in range(len(args.devices.split(","))))]
    env["FIDESLIB_AUX_POLY_CACHE_LIMIT"] = "0"
    env["FIDESLIB_USE_GRAPH_CAPTURE"] = "0"
    env.setdefault("OMP_NUM_THREADS", "4")
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv, "--worker"],
                           env=env, text=True, capture_output=True)
    print(child.stdout, end="")
    print(child.stderr, file=sys.stderr, end="")
    return 0 if child.returncode == 0 and MARKER in child.stdout.splitlines() else 1


if __name__ == "__main__":
    sys.exit(main())
