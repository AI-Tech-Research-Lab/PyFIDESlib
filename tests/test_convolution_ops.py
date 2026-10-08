"""Numerical GPU tests for hoisted rotations, plaintext dot products and convolution.

    python tests/test_convolution_ops.py
    WSUM_SCALING=FIXEDMANUAL python tests/test_convolution_ops.py
    CUDA_VISIBLE_DEVICES=0,1 WSUM_DEVICES=0,1 python tests/test_convolution_ops.py

Also runs the inherited weighted-sum regression tests. Respects CUDA_VISIBLE_DEVICES.
"""

import unittest

from test_linear_wsum import LinearWSumTest, fhe, make_context


class ConvolutionOpsTest(LinearWSumTest):
    rotation_indices = list(range(-3, 0)) + list(range(1, 16))

    def setUp(self):
        super().setUp()
        self.cc.SetPlaintextCache(None)

    def encode(self, values, level=0, degree=1):
        return self.cc.MakeCKKSPackedPlaintext(values, noiseScaleDeg=degree, level=level)

    def test_hoisted_rotations_preserve_order_and_input(self):
        ct = self.cts[0]
        x = [row[0] for row in self.values]
        indices = [1, -1, 0, 2, 1, 16]
        self.cc.SetCiphertextCache(0)
        ct.Offload()
        results = self.cc.EvalFastRotation(ct, indices)
        self.assertEqual(len(results), len(indices))
        for index, result in zip(indices, results):
            self.assert_values(result, [x[(i + index) % 16] for i in range(16)])
            self.assertEqual(result.GetLevel(), ct.GetLevel())
            self.assertEqual(result.GetNoiseScaleDeg(), ct.GetNoiseScaleDeg())
        self.cc.EvalAddInPlace(results[2], 1.0)
        self.assert_values(ct, x)
        self.assert_values(results[-1], x)
        self.assertEqual(self.cc.EvalFastRotation(ct, []), [])
        self.assert_values(self.cc.EvalFastRotation(ct, [0, 0])[0], x)

    def test_plaintext_dot_product_alias_cache_and_chaining(self):
        weights = [[(i + 1) / 10, -0.2 * (i % 3), 0.5] for i in range(16)]
        pts = [self.encode([row[k] for row in weights]) for k in range(3)]
        self.cc.SetCiphertextCache(0)
        self.cc.SetPlaintextCache(0)
        for ct in self.cts:
            ct.Offload()
        out = self.cts[0]
        alias = out
        level = out.GetLevel()
        expected = [sum(a * b for a, b in zip(values, w)) for values, w in zip(self.values, weights)]
        self.assertIsNone(out.dotProductPt(self.cts, pts))
        self.assertEqual(out.GetNoiseScaleDeg(), 2)
        self.assertEqual(out.GetLevel(), level)
        self.assert_values(alias, expected)
        self.cc.RescaleInPlace(out)
        self.assert_values(out, expected)
        self.assertEqual(out.GetLevel(), level + 1)
        self.assert_values(self.cts[1], [row[1] for row in self.values])
        pt = self.encode([0.25] * 16, level=out.GetLevel())
        out.dotProductPt([out], [pt])
        self.cc.RescaleInPlace(out)
        self.assert_values(out, [0.25 * value for value in expected])

    def test_convolution_matches_plaintext_formula(self):
        x = [row[0] for row in self.values]
        for g_step, indexes, stride in [(1, [0], 1), (2, [-1, 0], 2),
                                        (4, [-2, 0, 1], 1), (8, [0], -1), (16, [0], 1)]:
            with self.subTest(g_step=g_step, indexes=indexes, stride=stride):
                b_step = len(indexes)
                weights = [[(i + 2 * j - k) / 100 for i in range(16)]
                           for j in range(g_step) for k in range(b_step)]
                pts = [self.encode(w) for w in weights]
                out = self.cts[0].Clone()
                alias = out
                self.cc.SetCiphertextCache(0)
                self.cc.SetPlaintextCache(0)
                out.Offload()
                self.assertIsNone(self.cc.ConvolutionTransformInPlace(
                    out, g_step, b_step, pts, indexes, stride=stride))
                expected = [sum(weights[j * b_step + k][(i + stride * (g_step - j)) % 16]
                                * x[(i + stride * (g_step - j) + index) % 16]
                                for j in range(g_step) for k, index in enumerate(indexes))
                            for i in range(16)]
                self.assertEqual(out.GetNoiseScaleDeg(), 2)
                self.assertEqual(out.GetLevel(), self.cts[0].GetLevel())
                self.assert_values(alias, expected)
                self.cc.RescaleInPlace(out)
                self.assert_values(out, expected)
                self.assert_values(self.cts[0], x)

    def test_convolution_degree_two_input(self):
        out = self.cts[0].Clone()
        out.evalLinearWSumMutable(1, [self.cts[0]], [2.0])
        level = out.GetLevel()
        pts = [self.encode([0.5] * 16, level=level + 1)]
        self.cc.ConvolutionTransformInPlace(out, 1, 1, pts, [0], stride=0)
        self.assertEqual(out.GetLevel(), level + 1)
        self.assertEqual(out.GetNoiseScaleDeg(), 2)
        self.cc.RescaleInPlace(out)
        self.assert_values(out, [row[0] for row in self.values])

    def test_new_validation(self):
        out = self.cts[0].Clone()
        pt = self.encode([0.5] * 16)
        out.SetSlots(32)
        with self.assertRaisesRegex(ValueError, "rotation key"):
            self.cc.EvalFastRotation(out, [16])
        out.SetSlots(16)
        for ctxs, pts in [([], []), ([out], []), ([None], [pt]), ([out], [None]),
                         ([out], [self.encode([0.5] * 16, level=1)]),
                         ([out], [self.encode([0.5] * 16, degree=2)])]:
            with self.subTest(ctxs=ctxs, pts=pts), self.assertRaises(ValueError):
                out.dotProductPt(ctxs, pts)
        degree2 = out.Clone()
        degree2.evalLinearWSumMutable(1, [out], [1])
        with self.assertRaisesRegex(ValueError, "degree 1"):
            out.dotProductPt([degree2], [pt])
        other_key = self.encrypt([0.1] * 16, self.other_keys)
        with self.assertRaisesRegex(ValueError, "same key"):
            out.dotProductPt([out, other_key], [pt, pt])
        for args in [(0, 1, [pt], [0]), (6, 1, [pt] * 6, [0]),
                     (1, 2, [pt, pt], [0]), (2, 1, [pt], [0]),
                     (1, 1, [None], [0]), (1, 1, [self.encode([0.5] * 16, level=1)], [0])]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.cc.ConvolutionTransformInPlace(out, *args)
        with self.assertRaisesRegex(ValueError, "rowSize"):
            self.cc.ConvolutionTransformInPlace(out, 1, 1, [pt], [0], rowSize=2)
        self.assert_values(out, [row[0] for row in self.values])

    def test_new_context_checks(self):
        other = make_context(devices=())
        keys = other.KeyGen()
        pt = other.MakeCKKSPackedPlaintext([0.5] * 16)
        ct = other.Encrypt(keys.publicKey, pt)
        for operation in [lambda: self.cc.EvalFastRotation(ct, [1]),
                          lambda: self.cts[0].dotProductPt([self.cts[0]], [pt]),
                          lambda: self.cc.ConvolutionTransformInPlace(self.cts[0], 1, 1, [pt], [0])]:
            with self.assertRaisesRegex(ValueError, "context"):
                operation()
        for operation in [lambda: other.EvalFastRotation(ct, [1]),
                          lambda: ct.dotProductPt([ct], [pt]),
                          lambda: other.ConvolutionTransformInPlace(ct, 1, 1, [pt], [0])]:
            with self.assertRaisesRegex(RuntimeError, "GPU context"):
                operation()

    def test_rotation_helper_includes_boundaries(self):
        self.assertEqual(fhe.convolution_rotation_indices(1, 1, [0]), [1])
        self.assertEqual(fhe.convolution_rotation_indices(2, 2, [-1, 0], stride=2), [-1, 2, 4])
        self.assertEqual(fhe.convolution_rotation_indices(16, 1, [0]), list(range(1, 9)))
        for g_step in (0, 6):
            with self.assertRaises(ValueError):
                fhe.convolution_rotation_indices(g_step, 1, [0])


if __name__ == "__main__":
    # Imported base class remains useful as fixtures, but run only the extended suite.
    unittest.main(defaultTest="ConvolutionOpsTest")
