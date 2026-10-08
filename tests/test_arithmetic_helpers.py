"""CPU/GPU checks for the additional arithmetic bindings, plus weighted-sum regression.

    python tests/test_arithmetic_helpers.py
    WSUM_SCALING=FIXEDMANUAL python tests/test_arithmetic_helpers.py
    CUDA_VISIBLE_DEVICES=0,1 WSUM_DEVICES=0,1 python tests/test_arithmetic_helpers.py
"""

import unittest

from test_linear_wsum import LinearWSumTest, make_context


class ArithmeticHelpersTest(LinearWSumTest):
    multiplication_keys = True

    def test_add_many_preserves_inputs_and_handles_singleton(self):
        self.cc.SetCiphertextCache(0)
        for ct in self.cts:
            ct.Offload()
        for indices in ([0], [0, 1], [0, 1, 2], [0, 1, 2, 0, 1]):
            with self.subTest(indices=indices):
                out = self.cc.EvalAddMany([self.cts[k] for k in indices])
                self.assert_values(out, [sum(row[k] for k in indices) for row in self.values])
                self.cc.EvalAddInPlace(out, 1.0)
                for k, ct in enumerate(self.cts):
                    self.assert_values(ct, [row[k] for row in self.values])

    def test_add_many_in_place_preserves_first_object_identity(self):
        self.cc.SetCiphertextCache(0)
        for count in (1, 3, 5):
            with self.subTest(count=count):
                cts = [self.cts[k % 3].Clone() for k in range(count)]
                alias = cts[0]
                for ct in cts:
                    ct.Offload()
                self.assertIsNone(self.cc.EvalAddManyInPlace(cts))
                self.assertIs(alias, cts[0])
                self.assert_values(alias, [sum(row[k % 3] for k in range(count)) for row in self.values])

    def test_square_and_negate_in_place(self):
        self.cc.SetCiphertextCache(0)
        x = [row[0] for row in self.values]
        out = self.cts[0].Clone()
        alias = out
        out.Offload()
        reference = self.cc.EvalSquare(self.cts[0])
        self.assertIsNone(self.cc.EvalSquareInPlace(out))
        self.assertEqual(out.GetLevel(), reference.GetLevel())
        self.assertEqual(out.GetNoiseScaleDeg(), reference.GetNoiseScaleDeg())
        self.assert_values(alias, [v * v for v in x])
        self.cc.RescaleInPlace(out)
        reference = self.cc.EvalNegate(out)
        self.assertIsNone(self.cc.EvalNegateInPlace(out))
        self.assertEqual(out.GetLevel(), reference.GetLevel())
        self.assertEqual(out.GetNoiseScaleDeg(), reference.GetNoiseScaleDeg())
        self.assert_values(alias, [-v * v for v in x])
        self.assert_values(self.cts[0], x)

    def test_level_drop_preserves_value_and_alias(self):
        out = self.cts[0].Clone()
        alias = out
        level = out.GetLevel()
        self.assertIsNone(self.cc.SetLevel(out, level))
        out.Offload()
        self.assertIsNone(self.cc.SetLevel(out, level + 2))
        self.assertEqual(out.GetLevel(), level + 2)
        self.assert_values(alias, [row[0] for row in self.values])
        with self.assertRaises(ValueError):
            self.cc.SetLevel(out, level)
        with self.assertRaises(ValueError):
            self.cc.SetLevel(out, 7)
        self.assert_values(alias, [row[0] for row in self.values])

    def test_argument_validation(self):
        for method in (self.cc.EvalAddMany, self.cc.EvalAddManyInPlace):
            for cts in ([], [None], [self.cts[0], None]):
                with self.subTest(method=method, cts=cts), self.assertRaises(ValueError):
                    method(cts)
        with self.assertRaisesRegex(ValueError, "repeated"):
            self.cc.EvalAddManyInPlace([self.cts[0], self.cts[0]])
        other_key = self.encrypt([0.1] * 16, self.other_keys)
        for method in (self.cc.EvalAddMany, self.cc.EvalAddManyInPlace):
            with self.assertRaisesRegex(ValueError, "same key"):
                method([self.cts[0], other_key])
        for method in (self.cc.EvalSquareInPlace, self.cc.EvalNegateInPlace):
            with self.assertRaises(ValueError):
                method(None)
        with self.assertRaises(ValueError):
            self.cc.SetLevel(None, 1)

    def test_cpu_backend_and_context_validation(self):
        cc = make_context(devices=())
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        x = [row[0] for row in self.values]
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))

        def check(result, expected):
            pt = cc.Decrypt(keys.secretKey, result)
            pt.SetLength(16)
            self.assertLess(max(abs(a - b) for a, b in zip(pt.GetRealPackedValue(), expected)), 1e-5)

        single = cc.EvalAddMany([ct])
        cc.EvalAddInPlace(single, 1.0)
        check(ct, x)
        check(single, [v + 1 for v in x])
        check(cc.EvalAddMany([ct, ct, ct]), [3 * v for v in x])
        cts = [ct.Clone() for _ in range(3)]
        alias = cts[0]
        self.assertIsNone(cc.EvalAddManyInPlace(cts))
        check(alias, [3 * v for v in x])
        square = ct.Clone()
        cc.EvalSquareInPlace(square)
        check(square, [v * v for v in x])
        negative = ct.Clone()
        cc.EvalNegateInPlace(negative)
        check(negative, [-v for v in x])
        dropped = ct.Clone()
        cc.SetLevel(dropped, 2)
        self.assertEqual(dropped.GetLevel(), 2)
        check(dropped, x)
        check(ct, x)
        for operation in (lambda: self.cc.EvalAddMany([ct]),
                          lambda: self.cc.EvalAddManyInPlace([ct]),
                          lambda: self.cc.EvalSquareInPlace(ct),
                          lambda: self.cc.EvalNegateInPlace(ct),
                          lambda: self.cc.SetLevel(ct, 1)):
            with self.assertRaisesRegex(ValueError, "context"):
                operation()


if __name__ == "__main__":
    unittest.main(defaultTest="ArithmeticHelpersTest")
