"""FE functional bootstrapping: Python API, CPU/GPU parity and handle ownership.

    OMP_NUM_THREADS=8 python tests/test_functional_bootstrap.py
    CUDA_VISIBLE_DEVICES=2,3 PYFIDES_TEST_DEVICES=0,1 OMP_NUM_THREADS=8 \
        python tests/test_functional_bootstrap.py

Each numerical case gets its own process: bootstrap precomputations are shared.
The small ring and HEStd_NotSet are for tests only.
"""

import cmath
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fideslib_py as fhe

DEVICES = [int(d) for d in os.environ.get("PYFIDES_TEST_DEVICES", "0").split(",")]
# Exact Fourier series: cos(pi*x) + 0.25*cos(5*pi*x), and sin(pi*x).
# Real lists exercise conversion to vector<complex<double>> as in openfhe-python.
COEFFICIENTS = [0, 0.5, 0, 0, 0, 0.125]
SINE = [0, -0.5j]


def context(scaling="FLEXIBLEAUTO", distribution="SPARSE_ENCAPSULATED", *, slots=8,
            budget=(3, 2), complex_data=False, first_mod=60):
    skd = getattr(fhe, distribution)
    depth = fhe.CryptoContext.GetFEFBTDepth(
        levelBudget=list(budget), coefficients=COEFFICIENTS, skd=skd, firstModSize=first_mod) + 6
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_NotSet)
    params.SetRingDim(4096)
    params.SetMultiplicativeDepth(depth)
    params.SetScalingModSize(59)
    params.SetFirstModSize(first_mod)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(slots)
    params.SetScalingTechnique(getattr(fhe, scaling))
    params.SetSecretKeyDist(skd)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetCKKSDataType(fhe.COMPLEX if complex_data else fhe.REAL)
    params.SetDevices([])
    cc = fhe.GenCryptoContext(params)
    for feature in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(feature)
    return cc, depth


def check(cc, keys, ct, expected):
    pt = cc.Decrypt(keys.secretKey, ct)
    pt.SetLength(len(expected))
    actual = pt.GetCKKSPackedValue()
    error = max(abs(a - b) for a, b in zip(actual, expected))
    assert error < 1e-7, f"function error {error:.3e}"


def series(coefficients, x):
    return [2 * sum(c * cmath.exp(1j * math.pi * j * v.real)
                    for j, c in enumerate(coefficients)).real for v in x]


def numerical_case(scaling, distribution, packing):
    slots = 2048 if packing == "full" else 8
    budget = [1, 1] if packing == "full" else [3, 2]
    cc, depth = context(scaling, distribution, slots=slots, budget=budget,
                        complex_data=packing == "complex")
    # Keyword names, slots=0 for full packing, and default dim1 match the official API.
    cc.EvalFEFuncBootstrapSetup(levelBudget=budget, slots=0 if packing == "full" else slots)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    x = [-0.45, -0.3, -0.15, 0.0, 0.1, 0.2, 0.35, 0.45]
    if packing == "complex":
        x = [v + 0.1j for v in x]
    level = depth - (budget[1] + 1)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x, level=level, slots=slots))
    with tempfile.TemporaryDirectory() as temp:
        def snapshot(ciphertext):
            filename = str(Path(temp) / "ciphertext.json")
            assert fhe.SerializeToFile(filename, ciphertext, fhe.JSON)
            data = json.loads(Path(filename).read_text())["value0"]["ptr_wrapper"]["data"]
            data.pop("value0")  # context/key pointer identities
            data["m"] = data["m"]["ptr_wrapper"]["data"]
            data["v"] = [(p["f"], [(limb["f"], limb["v"]["ptr_wrapper"]["data"]["v"])
                                    for limb in p["v"]]) for p in data["v"]]
            return data

        original = snapshot(ct)
        cpu = cc.EvalFEFuncBootstrap(ciphertext=ct, coefficients=COEFFICIENTS)
        reference = snapshot(cpu)
        check(cc, keys, cpu, series(COEFFICIENTS, x))
        cpu_powers = cc.EvalFEFuncBootstrapPrecompute(ciphertext=ct, coefficients=COEFFICIENTS)
        assert isinstance(cpu_powers, fhe.SeriesPowers)
        references = []
        # Reuse powers, evaluate a short complex series, and accept trailing zeros.
        functions = [COEFFICIENTS, SINE, COEFFICIENTS + [0j, 0j]]
        for coefficients in functions:
            out = cc.EvalFEFuncBootstrapWithPrecomp(powers=cpu_powers, coefficients=coefficients)
            check(cc, keys, out, series(coefficients, x))
            references.append(snapshot(out))
        assert snapshot(ct) == original
        cc.SetDevices(DEVICES)
        cc.SetBootstrapCache(0)
        cc.SetRotationKeyCache(0)
        cc.LoadContext(keys.publicKey)
        cc.SetCiphertextCache(0)
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "built on the CPU"):
            cc.EvalFEFuncBootstrapWithPrecomp(cpu_powers, COEFFICIENTS)
        ct.Offload()
        gpu = cc.EvalFEFuncBootstrap(ciphertext=ct, coefficients=COEFFICIENTS)
        assert snapshot(gpu) == reference, "CPU/GPU RNS coefficients or metadata differ"
        check(cc, keys, gpu, series(COEFFICIENTS, x))
        assert gpu.GetLevel() < level
        ct.Offload()
        powers = cc.EvalFEFuncBootstrapPrecompute(ciphertext=ct, coefficients=COEFFICIENTS)
        for _ in range(2):
            for coefficients, expected in zip(functions, references):
                out = cc.EvalFEFuncBootstrapWithPrecomp(powers=powers, coefficients=coefficients)
                out.Offload()
                assert snapshot(out) == expected, "shared CPU/GPU powers differ"
                check(cc, keys, out, series(coefficients, x))
        assert snapshot(ct) == original, "input mutated"
        assert cc.GetBootstrapCacheLoadedBytes() > 0
        # Destroy all ordinary context owners before the last GPU powers handle.
        del out, gpu, ct, cpu, cpu_powers, cc, keys
        gc.collect()
        del powers
        gc.collect()
    print(f"PASS {scaling}:{distribution}:{packing}", flush=True)


class FunctionalBootstrapTest(unittest.TestCase):
    def test_cpu_gpu_parity(self):
        cases = [(s, "SPARSE_ENCAPSULATED", "sparse") for s in
                 ("FLEXIBLEAUTO", "FLEXIBLEAUTOEXT", "FIXEDAUTO", "FIXEDMANUAL")]
        cases += [("FLEXIBLEAUTO", d, "sparse") for d in ("UNIFORM_TERNARY", "SPARSE_TERNARY")]
        cases += [("FLEXIBLEAUTO", "SPARSE_ENCAPSULATED", p) for p in ("full", "complex")]
        for case in cases:
            with self.subTest(case=case):
                result = subprocess.run([sys.executable, __file__, "--case", *case],
                                        capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])

    def test_validation(self):
        cc, depth = context()
        other, _ = context(slots=16)
        keys = cc.KeyGen()
        other_keys = other.KeyGen()
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([0.1], level=depth - 3, slots=8))
        foreign = other.Encrypt(other_keys.publicKey, other.MakeCKKSPackedPlaintext([0.1], slots=16))
        for name in ("EvalFEFuncBootstrap", "EvalFEFuncBootstrapPrecompute"):
            for bad in (None, foreign):
                with self.assertRaises(ValueError):
                    getattr(cc, name)(bad, COEFFICIENTS)
        with self.assertRaises(RuntimeError):
            cc.EvalFEFuncBootstrapWithPrecomp(None, COEFFICIENTS)
        with self.assertRaises(TypeError):
            fhe.SeriesPowers()
        for budget, dim1, slots in (([], [0, 0], 8), ([3, 2], [], 8), ([3, 2], [0, 0], 3)):
            with self.assertRaises(RuntimeError):
                cc.EvalFEFuncBootstrapSetup(budget, dim1, slots)
        cc.EvalFEFuncBootstrapSetup([3, 2], [0, 0], 8)
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalBootstrapKeyGen(keys.secretKey, 8)
        for coefficients in ([], [0.25], [0.25, 0], [0, float("nan")], [0, 1e-20]):
            for name in ("EvalFEFuncBootstrap", "EvalFEFuncBootstrapPrecompute"):
                with self.assertRaises(ValueError):
                    getattr(cc, name)(ct, coefficients)
        with self.assertRaises(RuntimeError):
            cc.EvalFEFuncBootstrapPrecompute(ct, SINE)
        powers = cc.EvalFEFuncBootstrapPrecompute(ct, COEFFICIENTS)
        with self.assertRaisesRegex(RuntimeError, "different CryptoContext"):
            other.EvalFEFuncBootstrapWithPrecomp(powers, COEFFICIENTS)
        # Standard bootstrapping can share the slot count with FE-FBT; StC-first still cannot,
        # since it would replace the standard setup.
        cc.EvalBootstrapSetup([3, 2], [0, 0], 8)
        with self.assertRaisesRegex(RuntimeError, "setup conflict"):
            cc.EvalBootstrapSetup([3, 2], [0, 0], 8, btsfirstboot=True)
        for budget, coefficients in (([], COEFFICIENTS), ([3, 2], [])):
            with self.assertRaises(RuntimeError):
                fhe.CryptoContext.GetFEFBTDepth(budget, coefficients)
        invalid, _ = context(first_mod=59)
        with self.assertRaises(RuntimeError):
            invalid.EvalFEFuncBootstrapSetup([3, 2], [0, 0], 8)
        cc.SetDevices(DEVICES)
        cc.LoadContext(keys.publicKey)
        # The standard setup came after EvalBootstrapKeyGen, so the device has no keys for it.
        with self.assertRaisesRegex(RuntimeError, "EvalBootstrapKeyGen"):
            cc.EvalBootstrap(ct)
        with self.assertRaisesRegex(RuntimeError, "already loaded"):
            cc.EvalFEFuncBootstrapSetup([3, 2], [0, 0], 8)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--case":
        numerical_case(*sys.argv[2:])
    else:
        unittest.main()
