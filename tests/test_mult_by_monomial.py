"""MultByMonomial / MultByMonomialInPlace on the CPU and GPU backends.

    CUDA_VISIBLE_DEVICES=3 python3 tests/test_mult_by_monomial.py
    CUDA_VISIBLE_DEVICES=2,3 PYFIDES_TEST_DEVICES=0,1 python3 tests/test_mult_by_monomial.py

Parameters are small, non-secure test parameters.
"""

import json
from pathlib import Path
import tempfile
import unittest

from test_upstream_220 import DEVICES, check, context, fhe

X = [0.2 + 0.1j, -0.3 + 0.2j, 0.4 - 0.3j, -0.1j]


def metadata(ct):
    return ct.GetLevel(), ct.GetNoiseScaleDeg(), ct.GetScalingFactor()


def snapshot(ct):
    """RNS coefficients and metadata, without cereal's shared-pointer identity graph."""
    with tempfile.TemporaryDirectory() as temp:
        filename = str(Path(temp) / "ciphertext.json")
        assert fhe.SerializeToFile(filename, ct, fhe.JSON)
        data = json.loads(Path(filename).read_text())["value0"]["ptr_wrapper"]["data"]
    data.pop("value0")  # context/key object graph
    data["m"] = data["m"]["ptr_wrapper"]["data"]
    data["v"] = [(poly["f"], [(limb["f"], limb["v"]["ptr_wrapper"]["data"]["v"]) for limb in poly["v"]])
                 for poly in data["v"]]
    return data


class MultByMonomialTest(unittest.TestCase):
    def test_slot_factors_and_metadata(self):
        for devices in ([], DEVICES):
            for scaling in ("FLEXIBLEAUTO", "FIXEDMANUAL"):
                with self.subTest(devices=devices, scaling=scaling):
                    cc, keys = context(devices=devices, complex_data=True, scaling=scaling)
                    if devices:
                        cc.LoadContext(keys.publicKey)
                    n = cc.GetRingDimension()
                    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(X))
                    before = metadata(ct)
                    for power, factor in ((0, 1), (n // 2, 1j), (n, -1), (3 * n // 2, -1j),
                                          (2 * n + n // 2, 1j), (4 * n - n // 2, -1j)):
                        with self.subTest(power=power):
                            result = cc.MultByMonomial(ct, power)
                            check(cc, keys, result, [factor * v for v in X])
                            self.assertEqual(metadata(result), before)
                    check(cc, keys, ct, X)  # the input is untouched

                    alias = ct
                    self.assertIsNone(cc.MultByMonomialInPlace(ct, n // 2))
                    check(cc, keys, alias, [1j * v for v in X])
                    self.assertEqual(metadata(alias), before)

                    # A lower level rebuilds the GPU's cached monomial; i * i = -1 composes exactly.
                    low = cc.EvalMult(cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(X)), 1.0)
                    if scaling == "FIXEDMANUAL":
                        cc.RescaleInPlace(low)
                    cc.MultByMonomialInPlace(low, n // 2)
                    cc.MultByMonomialInPlace(low, n // 2)
                    check(cc, keys, low, [-v for v in X])

                    if devices:  # offloaded input is reloaded first
                        off = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(X))
                        off.Offload()
                        check(cc, keys, cc.MultByMonomial(off, 3 * n // 2), [-1j * v for v in X])

    def test_cpu_gpu_bit_exact(self):
        cc, keys = context(devices=[], complex_data=True)
        n = cc.GetRingDimension()
        cts = [cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(X)) for _ in range(2)]
        cts[1] = cc.EvalMult(cts[1], 1.0)
        powers = (n // 2, 3 * n // 2, 7)
        reference = [snapshot(cc.MultByMonomial(ct, p)) for ct in cts for p in powers]
        cc.SetDevices(DEVICES)
        cc.LoadContext(keys.publicKey)
        for i, (ct, p) in enumerate((ct, p) for ct in cts for p in powers):
            with self.subTest(level=ct.GetLevel(), power=p):
                self.assertEqual(snapshot(cc.MultByMonomial(ct, p)), reference[i])

    def test_rejects_foreign_ciphertext(self):
        cc, keys = context(devices=[], complex_data=True)
        other, other_keys = context(devices=[], complex_data=True)
        ct = other.Encrypt(other_keys.publicKey, other.MakeCKKSPackedPlaintext(X))
        with self.assertRaises(ValueError):
            cc.MultByMonomial(ct, 1)
        with self.assertRaises(ValueError):
            cc.MultByMonomialInPlace(ct, 1)


if __name__ == "__main__":
    unittest.main()
