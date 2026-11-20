"""OpenFHE's EvalSum, EvalSumRows and EvalSumCols on the CPU fallback and on the GPU.

    CUDA_VISIBLE_DEVICES=3 python3 tests/test_eval_sum.py
    CUDA_VISIBLE_DEVICES=2,3 PYFIDES_TEST_DEVICES=0,1 python3 tests/test_eval_sum.py

FIDESlib's own test suite checks that the GPU results are bit-identical to OpenFHE; here the
results are checked against plain sums. Parameters are small, non-secure test parameters.
"""

import unittest

from test_upstream_220 import DEVICES, check, context, decode

# (ring dimension, slots): sparse and fully packed.
SHAPES = ((8192, 16), (1024, 512))


def values(slots):
    return [((i * 37) % 17) / 17 - 0.5 for i in range(slots)]


def plain_sum(x, size):
    """EvalSum(ct, size): slot i receives x[i] + ... + x[i + size - 1], cyclic."""
    n = len(x)
    return [sum(x[(i + k) % n] for k in range(size)) for i in range(n)]


def plain_sum_rows(x, row_size, ring):
    """EvalSumRows(ct, rowSize) sums over the whole ring, (ring/2)/rowSize terms, as OpenFHE does;
    sparse packing repeats the batch, so each column sum is scaled by (ring/2)/slots."""
    n = len(x)
    return [sum(x[(i + k * row_size) % n] for k in range(ring // 2 // row_size)) for i in range(n)]


def plain_sum_cols(x, num_cols):
    """EvalSumCols(ct, numCols): every slot of a block of numCols slots receives the block's sum."""
    return [sum(x[i - i % num_cols:i - i % num_cols + num_cols]) for i in range(len(x))]


def metadata(ct):
    return ct.GetLevel(), ct.GetNoiseScaleDeg(), ct.GetSlots()


def sum_context(ring, slots, scaling, devices, row_size):
    cc, keys = context(devices=devices, ring=ring, slots=slots, scaling=scaling)
    cc.EvalSumKeyGen(keys.secretKey)
    cc.EvalSumRowsKeyGen(keys.secretKey, row_size)
    cc.EvalSumColsKeyGen(keys.secretKey)
    if devices:
        cc.LoadContext(keys.publicKey)
    return cc, keys


class EvalSumTest(unittest.TestCase):
    def test_values_cpu_and_gpu(self):
        row_size, num_cols = 4, 8
        for ring, slots in SHAPES:
            for devices in ([], DEVICES):
                for scaling in ("FLEXIBLEAUTO", "FIXEDMANUAL"):
                    with self.subTest(ring=ring, slots=slots, devices=devices, scaling=scaling):
                        cc, keys = sum_context(ring, slots, scaling, devices, row_size)
                        x = values(slots)
                        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
                        before = metadata(ct)

                        total = cc.EvalSum(ct, slots)
                        rows = cc.EvalSumRows(ct, row_size)
                        cols = cc.EvalSumCols(ct, num_cols)

                        # Column sums over the whole ring grow with ring/slots; scale the bound.
                        check(cc, keys, total, plain_sum(x, slots), 1e-4)
                        check(cc, keys, rows, plain_sum_rows(x, row_size, ring), 1e-4 * ring / slots)
                        check(cc, keys, cols, plain_sum_cols(x, num_cols), 1e-4)
                        check(cc, keys, ct, x)  # the input is preserved
                        self.assertEqual(metadata(ct), before)
                        # The folds consume no level; EvalSumCols' mask multiplication leaves a
                        # pending rescale (degree 2), as in OpenFHE.
                        self.assertEqual(metadata(total), before)
                        self.assertEqual(metadata(rows), before)
                        self.assertEqual(cols.GetNoiseScaleDeg(), 2)

    def test_after_multiplication(self):
        ring, slots, num_cols = 8192, 16, 4
        for devices in ([], DEVICES):
            with self.subTest(devices=devices):
                cc, keys = sum_context(ring, slots, "FLEXIBLEAUTO", devices, 2)
                x = values(slots)
                ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
                squared = cc.EvalMult(ct, ct)
                x2 = [v * v for v in x]
                check(cc, keys, cc.EvalSum(squared, slots), plain_sum(x2, slots), 1e-4)
                cols = cc.EvalSumCols(squared, num_cols)
                check(cc, keys, cols, plain_sum_cols(x2, num_cols), 1e-4)
                # The result keeps working: one more multiplication.
                check(cc, keys, cc.EvalMult(cols, cols), [v * v for v in plain_sum_cols(x2, num_cols)], 1e-4)

    def test_partial_sums(self):
        ring, slots = 8192, 16
        cc, keys = sum_context(ring, slots, "FLEXIBLEAUTO", DEVICES, 4)
        x = values(slots)
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
        # Batch sizes below the context's batch size reuse its keys (a subset of the fold). As in
        # OpenFHE, the radix-4 fold sums the next power of two terms: size 5 sums 8 (1, 2, 3, then
        # 4 of the accumulator), on the CPU fallback and on the GPU alike.
        for size, terms in ((2, 2), (4, 4), (8, 8), (5, 8)):
            with self.subTest(size=size):
                actual = decode(cc, keys, cc.EvalSum(ct, size), size)
                expected = plain_sum(x, terms)[:size]
                self.assertLess(max(abs(a - b) for a, b in zip(actual, expected)), 1e-4)

    def test_errors(self):
        ring, slots = 8192, 16
        cc, keys = context(devices=DEVICES, ring=ring, slots=slots)
        with self.assertRaisesRegex(RuntimeError, "rowSize"):
            cc.EvalSumRowsKeyGen(keys.secretKey, 0)
        with self.assertRaisesRegex(RuntimeError, "subringDim"):
            cc.EvalSumRowsKeyGen(keys.secretKey, 4, 1024)
        cc.EvalSumKeyGen(keys.secretKey)
        cc.EvalSumRowsKeyGen(keys.secretKey, 4)
        cc.LoadContext(keys.publicKey)
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(values(slots)))

        # Missing keys raise before the GPU fold starts.
        with self.assertRaisesRegex(ValueError, "EvalSumColsKeyGen"):
            cc.EvalSumCols(ct, 4)
        with self.assertRaisesRegex(ValueError, "EvalSumRowsKeyGen"):
            cc.EvalSumRows(ct, 2)
        with self.assertRaisesRegex(RuntimeError, "numCols"):
            cc.EvalSumCols(ct, 0)
        with self.assertRaisesRegex(RuntimeError, "numCols"):
            cc.EvalSumCols(ct, slots + 1)
        with self.assertRaisesRegex(RuntimeError, "rowSize"):
            cc.EvalSumRows(ct, 0)
        with self.assertRaisesRegex(RuntimeError, "LoadContext"):
            cc.EvalSumColsKeyGen(keys.secretKey)
        with self.assertRaisesRegex(ValueError, "this context"):
            other, other_keys = context(devices=[], ring=ring, slots=slots)
            cc.EvalSum(other.Encrypt(other_keys.publicKey, other.MakeCKKSPackedPlaintext([1.0])), 2)
        # The context is still usable after the errors.
        check(cc, keys, cc.EvalSum(ct, slots), plain_sum(values(slots), slots), 1e-4)


if __name__ == "__main__":
    unittest.main()
