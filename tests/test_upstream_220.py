"""Numerical regressions for the FIDESlib 2.2.0 bindings.

    CUDA_VISIBLE_DEVICES=3 python tests/test_upstream_220.py
    CUDA_VISIBLE_DEVICES=2,3 PYFIDES_TEST_DEVICES=0,1 python tests/test_upstream_220.py

Bootstrap cases run in separate processes because precomputations are shared per context.
Parameters are small, non-secure test parameters.
"""

import json
import operator
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fideslib_py as fhe

DEVICES = [int(d) for d in os.environ.get("PYFIDES_TEST_DEVICES", "0").split(",")]


def context(*, devices=DEVICES, complex_data=False, depth=6, slots=16, ring=8192,
            scaling="FLEXIBLEAUTO", bootstrap=False):
    p = fhe.CCParams()
    p.SetSecurityLevel(fhe.HEStd_NotSet)
    p.SetRingDim(ring)
    p.SetMultiplicativeDepth(depth)
    p.SetScalingModSize(59 if bootstrap else 50)
    p.SetFirstModSize(60)
    p.SetNumLargeDigits(3)
    p.SetBatchSize(slots)
    p.SetScalingTechnique(getattr(fhe, scaling))
    p.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    p.SetKeySwitchTechnique(fhe.HYBRID)
    p.SetCKKSDataType(fhe.COMPLEX if complex_data else fhe.REAL)
    assert p.GetCKKSDataType() == (fhe.COMPLEX if complex_data else fhe.REAL)
    p.SetDevices(list(devices))
    cc = fhe.GenCryptoContext(p)
    for feature in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(feature)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    return cc, keys


def decode(cc, keys, ct, length):
    pt = cc.Decrypt(keys.secretKey, ct)
    pt.SetLength(length)
    return list(pt.GetCKKSPackedValue())


def check(cc, keys, ct, expected, tolerance=1e-5):
    actual = decode(cc, keys, ct, len(expected))
    assert len(actual) == len(expected)
    error = max(abs(a - b) for a, b in zip(actual, expected))
    assert error < tolerance, f"error {error:.3e} >= {tolerance:.3e}"


class Upstream220Test(unittest.TestCase):
    def test_mutable_cpu_gpu_and_aliases(self):
        for devices in ([], DEVICES):
            for scaling in ("FLEXIBLEAUTO", "FIXEDMANUAL"):
                with self.subTest(devices=devices, scaling=scaling):
                    cc, keys = context(devices=devices, scaling=scaling)
                    if devices:
                        cc.LoadContext(keys.publicKey)
                        cc.SetCiphertextCache(0)
                        cc.SetPlaintextCache(0)
                    x, y = [0.2, -0.3, 0.4, 0.1], [-0.1, 0.2, 0.3, 0.5]
                    def encrypt(v):
                        return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(v))
                    for name, op in (("Add", operator.add), ("Sub", operator.sub), ("Mult", operator.mul)):
                        expected = [op(a, b) for a, b in zip(x, y)]
                        for kind in ("ct", "pt-right", "pt-left", "in-place"):
                            with self.subTest(operation=name, kind=kind):
                                a, b = encrypt(x), encrypt(y)
                                # Exercise alignment of a degree-2 operand as well as offload/reload.
                                a = cc.EvalMult(a, 1.0)
                                if scaling == "FIXEDMANUAL":
                                    cc.RescaleInPlace(a)
                                a.Offload()
                                b.Offload()
                                if kind == "ct":
                                    result = getattr(cc, f"Eval{name}Mutable")(a, b)
                                elif kind == "in-place":
                                    alias = a
                                    self.assertIsNone(getattr(cc, f"Eval{name}MutableInPlace")(a, b))
                                    self.assertIs(a, alias)
                                    result = alias
                                else:
                                    pt = cc.MakeCKKSPackedPlaintext(y if kind == "pt-right" else x)
                                    result = getattr(cc, f"Eval{name}Mutable")(
                                        a if kind == "pt-right" else pt,
                                        pt if kind == "pt-right" else b)
                                check(cc, keys, result, expected)
                    a = encrypt(x)
                    a = cc.EvalMult(a, 1.0)
                    check(cc, keys, cc.EvalSquareMutable(a), [v * v for v in x])

    def test_complex_encoding_arithmetic_and_metadata(self):
        for devices in ([], DEVICES):
            with self.subTest(devices=devices):
                cc, keys = context(devices=devices, complex_data=True)
                if devices:
                    cc.LoadContext(keys.publicKey)
                x = [0.2 + 0.1j, -0.3 + 0.2j, 0.4 - 0.3j, -0.1j]
                pt = cc.MakeCKKSPackedPlaintext(x, slots=16)
                ct = cc.Encrypt(keys.publicKey, pt)
                check(cc, keys, ct, x)
                self.assertEqual(ct.GetSlots(), 16)
                self.assertGreater(ct.GetScalingFactor(), 0)
                check(cc, keys, cc.EvalSquare(ct), [v * v for v in x])
                check(cc, keys, cc.EvalMult(ct, pt), [v * v for v in x])
                # The original real-list overload must remain usable in a complex context.
                real = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([0.25] * 4))
                check(cc, keys, real, [0.25] * 4)

    def test_unload_and_serialization_preserve_latest_value(self):
        for devices in ([], DEVICES):
            with self.subTest(devices=devices):
                cc, keys = context(devices=devices)
                if devices:
                    cc.LoadContext(keys.publicKey)
                x = [0.2, -0.3, 0.4, 0.1]
                original = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
                ct = cc.EvalAdd(cc.EvalSquare(original), 0.125)
                expected = [v * v + 0.125 for v in x]
                with tempfile.TemporaryDirectory() as temp:
                    for sertype in (fhe.BINARY, fhe.JSON):
                        for offload in (False, True):
                            if offload:
                                ct.Offload()
                                self.assertIsNone(cc.UnloadCiphertext(ct))
                            filename = str(Path(temp) / "ciphertext")
                            self.assertTrue(fhe.SerializeToFile(filename, ct, sertype))
                            restored = cc.DeserializeCiphertextFromFile(filename, sertype)
                            check(cc, keys, restored, expected)
                            check(cc, keys, cc.EvalAdd(restored, 0.25), [v + 0.25 for v in expected])
                            check(cc, keys, original, x)
                    with self.assertRaises(RuntimeError):
                        cc.DeserializeCiphertextFromFile(str(Path(temp) / "absent"))
                    self.assertFalse(fhe.SerializeToFile(str(Path(temp) / "absent" / "ct"), ct))

    def test_negation_and_level_drop_preserve_scale_degree(self):
        cc, keys = context()
        cc.LoadContext(keys.publicKey)
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([0.2, -0.3]))
        ct = cc.EvalMult(ct, 2.0)
        state = (ct.GetLevel(), ct.GetNoiseScaleDeg(), ct.GetScalingFactor())
        neg = cc.EvalNegate(ct)
        self.assertEqual((neg.GetLevel(), neg.GetNoiseScaleDeg(), neg.GetScalingFactor()), state)
        check(cc, keys, neg, [-0.4, 0.6])
        cc.EvalNegateInPlace(neg)
        self.assertEqual((neg.GetLevel(), neg.GetNoiseScaleDeg(), neg.GetScalingFactor()), state)
        cc.SetLevel(neg, state[0] + 1)
        self.assertEqual(neg.GetNoiseScaleDeg(), state[1])
        self.assertEqual(neg.GetScalingFactor(), state[2])
        check(cc, keys, neg, [0.4, -0.6])

    def test_validation_of_new_bindings(self):
        cc, keys = context()
        cc.LoadContext(keys.publicKey)
        other, other_keys = context(devices=[])
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([0.2]))
        foreign = other.Encrypt(other_keys.publicKey, other.MakeCKKSPackedPlaintext([0.2]))
        foreign_pt = other.MakeCKKSPackedPlaintext([0.2])
        for name in ("Add", "Sub", "Mult"):
            for args in ((ct, None), (ct, foreign), (ct, foreign_pt), (foreign_pt, ct)):
                with self.assertRaises(ValueError):
                    getattr(cc, f"Eval{name}Mutable")(*args)
            with self.assertRaises(ValueError):
                getattr(cc, f"Eval{name}MutableInPlace")(ct, foreign)
        with self.assertRaises(ValueError):
            cc.UnloadCiphertext(foreign)
        with self.assertRaises(ValueError):
            cc.EvalSquareMutable(None)
        with self.assertRaises(ValueError):
            fhe.SerializeToFile("unused", None)

    def test_bootstrap_modes(self):
        cases = [f"{mode}:{scaling}:{iterations}"
                 for mode in ("standard", "stc")
                 for scaling in ("FLEXIBLEAUTO", "FLEXIBLEAUTOEXT", "FIXEDAUTO", "FIXEDMANUAL")
                 for iterations in (1, 2)] + ["lt:FLEXIBLEAUTO:1", "stc-lt:FLEXIBLEAUTO:2", "complex:FLEXIBLEAUTO:1"]
        for case in cases:
            with self.subTest(case=case):
                result = subprocess.run([sys.executable, __file__, "--bootstrap-case", case],
                                        capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])


def bootstrap_case(case):
    mode, scaling, iterations = case.split(":")
    iterations = int(iterations)
    stc, is_complex = mode.startswith("stc"), mode == "complex"
    budget = [1, 1] if "lt" in mode else [3, 3]
    slots, depth = (2048 if is_complex else 8), 30
    cc, keys = context(devices=[], complex_data=is_complex, depth=depth, slots=slots,
                       ring=4096, scaling=scaling, bootstrap=True)
    cc.EvalBootstrapSetup(budget, [0, 0], slots, btsfirstboot=stc)
    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    level = depth + 1 - (budget[1] + 2 + (scaling == "FLEXIBLEAUTOEXT")) if stc else depth - 1
    x = [0.1 + i / 100 for i in range(8)]
    if is_complex:
        x = [v + (0.02 - i / 100) * 1j for i, v in enumerate(x)]
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x, level=level, slots=slots))
    # Check the CPU replacement path and Python alias semantics before loading the GPU.
    cpu = ct.Clone()
    alias = cpu
    assert cc.EvalBootstrapInPlace(cpu, iterations, 12) is None
    check(cc, keys, alias, x, 1e-3)
    # Decryption adds randomized CKKS noise. Compare exact RNS coefficients and
    # metadata via JSON instead, excluding cereal's shared-pointer identity graph.
    temp = tempfile.TemporaryDirectory()
    def snapshot(result):
        filename = str(Path(temp.name) / "ciphertext.json")
        assert fhe.SerializeToFile(filename, result, fhe.JSON)
        data = json.loads(Path(filename).read_text())["value0"]["ptr_wrapper"]["data"]
        data.pop("value0")  # serialized context/key object graph, shared by both phases
        data["m"] = data["m"]["ptr_wrapper"]["data"]
        data["v"] = [(poly["f"], [(limb["f"], limb["v"]["ptr_wrapper"]["data"]["v"])
                                   for limb in poly["v"]]) for poly in data["v"]]
        return data
    reference = snapshot(alias)
    def matches_reference(result):
        assert snapshot(result) == reference, "CPU/GPU RNS coefficients or metadata differ"
        check(cc, keys, result, x, 1e-3)
    cc.SetDevices(DEVICES)
    cc.SetBootstrapCache(0)
    cc.SetRotationKeyCache(0)
    cc.LoadContext(keys.publicKey)
    # Dispatch, direct StC entry point, native in-place GPU path, and stage cache reloads.
    result = cc.EvalBootstrap(ct, iterations, 12)
    matches_reference(result)
    if stc:
        matches_reference(cc.EvalBootstrapStCFirst(ct, iterations, 12))
    else:
        try:
            cc.EvalBootstrapStCFirst(ct)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("standard precomputation accepted by StC-first")
    gpu = ct.Clone()
    alias = gpu
    gpu.Offload()
    assert cc.EvalBootstrapInPlace(gpu, iterations, 12) is None
    matches_reference(alias)
    assert gpu.GetLevel() < level
    assert cc.GetBootstrapCacheLoadedBytes() > 0
    check(cc, keys, ct, x)
    print(f"PASS {case}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--bootstrap-case":
        bootstrap_case(sys.argv[2])
    else:
        unittest.main()
