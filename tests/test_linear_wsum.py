"""GPU integration checks for Ciphertext.evalLinearWSumMutable.

Run with the Python used to build the module:
    python tests/test_linear_wsum.py
    WSUM_SCALING=FIXEDMANUAL python tests/test_linear_wsum.py
CUDA_VISIBLE_DEVICES selects the GPU; defaults to physical GPU 0.
"""

import math
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fideslib_py as fhe


def make_context(devices=None):
    if devices is None:
        devices = [int(device) for device in os.environ.get("WSUM_DEVICES", "0").split(",")]
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_NotSet)
    params.SetRingDim(1 << 14)
    params.SetMultiplicativeDepth(6)
    params.SetScalingModSize(50)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(16)
    params.SetScalingTechnique(getattr(fhe, os.environ.get("WSUM_SCALING", "FLEXIBLEAUTO")))
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetDevices(list(devices))
    cc = fhe.GenCryptoContext(params)
    for feature in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE):
        cc.Enable(feature)
    return cc


class LinearWSumTest(unittest.TestCase):
    rotation_indices = [-1, 1]
    multiplication_keys = False

    @classmethod
    def setUpClass(cls):
        cls.cc = make_context()
        cls.keys = cls.cc.KeyGen()
        cls.other_keys = cls.cc.KeyGen()
        if cls.multiplication_keys:
            cls.cc.EvalMultKeyGen(cls.keys.secretKey)
        cls.cc.EvalRotateKeyGen(cls.keys.secretKey, cls.rotation_indices)
        cls.cc.LoadContext(cls.keys.publicKey)
        cls.values = [[(i - 8) / 10, (i % 3) / 5, 0.3 - i / 20] for i in range(16)]

    def setUp(self):
        self.cc.SetCiphertextCache(None)
        self.cts = [self.encrypt([row[k] for row in self.values]) for k in range(3)]

    def encrypt(self, values, keys=None):
        return self.cc.Encrypt((keys or self.keys).publicKey, self.cc.MakeCKKSPackedPlaintext(values))

    def decoded(self, ct):
        pt = self.cc.Decrypt(self.keys.secretKey, ct)
        pt.SetLength(16)
        return list(pt.GetRealPackedValue())

    def assert_values(self, ct, expected):
        actual = self.decoded(ct)
        self.assertLess(max(abs(a - b) for a, b in zip(actual, expected)), 1e-5)

    def test_weighted_sum_preserves_inputs_and_rescales(self):
        out = self.cts[0].Clone()
        level = out.GetLevel()
        self.assertIsNone(out.evalLinearWSumMutable(3, self.cts, [2.0, -1.0, 0.5]))
        self.assertEqual(out.GetNoiseScaleDeg(), 2)
        self.assertEqual(out.GetLevel(), level)
        expected = [2 * a - b + 0.5 * c for a, b, c in self.values]
        self.assert_values(out, expected)
        self.cc.RescaleInPlace(out)
        self.assertEqual(out.GetNoiseScaleDeg(), 1)
        self.assertEqual(out.GetLevel(), level + 1)
        self.assert_values(out, expected)
        for k, ct in enumerate(self.cts):
            self.assert_values(ct, [row[k] for row in self.values])

    def test_alias_prefix_zero_weight_and_offload(self):
        out = self.cts[0]
        alias = out
        self.cc.SetCiphertextCache(0)
        for ct in self.cts:
            ct.Offload()
        # Only the first n entries are used; duplicate inputs and receiver aliasing work.
        out.evalLinearWSumMutable(3, [out, out, self.cts[1], None], [2.0, -0.5, 0.0, math.nan])
        self.cc.RescaleInPlace(out)
        self.assert_values(alias, [1.5 * row[0] for row in self.values])
        self.assert_values(self.cts[1], [row[1] for row in self.values])

    def test_convolution_and_chained_sum(self):
        ct = self.cts[0]
        ctxs = [self.cc.EvalRotate(ct, -1), ct, self.cc.EvalRotate(ct, 1)]
        out = ct.Clone()
        out.evalLinearWSumMutable(3, ctxs, [0.25, 0.5, 0.25])
        self.cc.RescaleInPlace(out)
        x = [row[0] for row in self.values]
        expected = [0.25 * x[(i - 1) % 16] + 0.5 * x[i] + 0.25 * x[(i + 1) % 16] for i in range(16)]
        self.assert_values(out, expected)
        # A lower-level destination can also combine inputs at different levels.
        out.evalLinearWSumMutable(2, [out, ct], [0.5, 0.25])
        self.cc.RescaleInPlace(out)
        self.assert_values(out, [0.5 * a + 0.25 * b for a, b in zip(expected, x)])

    def test_invalid_arguments_raise_without_mutation(self):
        out = self.cts[0].Clone()
        cases = [(0, [], []), (2, self.cts[:1], [1, 1]), (2, self.cts, [1]),
                 (1, [None], [1]), (1, self.cts, [math.inf]), (1, self.cts, [math.nan])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                out.evalLinearWSumMutable(*args)
        degree2 = self.cts[1].Clone()
        degree2.evalLinearWSumMutable(1, [self.cts[1]], [1])
        with self.assertRaisesRegex(ValueError, "degree 1"):
            out.evalLinearWSumMutable(1, [degree2], [1])
        self.cc.RescaleInPlace(degree2)
        with self.assertRaisesRegex(ValueError, "remaining levels"):
            out.evalLinearWSumMutable(1, [degree2], [1])
        other_key = self.encrypt([0.1] * 16, self.other_keys)
        with self.assertRaisesRegex(ValueError, "same key"):
            out.evalLinearWSumMutable(2, [self.cts[0], other_key], [1, 1])
        self.assert_values(out, [row[0] for row in self.values])

    def test_context_checks(self):
        cc = make_context(devices=())
        keys = cc.KeyGen()
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([0.2] * 16))
        with self.assertRaisesRegex(ValueError, "context"):
            self.cts[0].evalLinearWSumMutable(1, [ct], [1])
        with self.assertRaisesRegex(RuntimeError, "GPU context"):
            ct.evalLinearWSumMutable(1, [ct], [1])


if __name__ == "__main__":
    unittest.main()
