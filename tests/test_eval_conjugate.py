"""EvalConjugate on the GPU backend.

    CUDA_VISIBLE_DEVICES=3 python3 tests/test_eval_conjugate.py
    CUDA_VISIBLE_DEVICES=2,3 PYFIDES_TEST_DEVICES=0,1 python3 tests/test_eval_conjugate.py

The conjugation key comes from the bootstrapping setup. Parameters are small, non-secure test
parameters.
"""

import unittest

from test_upstream_220 import DEVICES, check, context

X = [1 + 2j, -3 + 0.5j, 0.25 - 0.75j, -0.5j]
Y = [0.5 - 0.25j, 0.2 + 0.1j, -1 + 0j, 0.3 + 0.4j]
SLOTS = 8
# Slot magnitudes above 1 need a looser bound than the shared 1e-5 default.
TOL = 1e-4


def conj(values):
    return [v.conjugate() for v in values]


def metadata(ct):
    return ct.GetLevel(), ct.GetNoiseScaleDeg(), ct.GetScalingFactor(), ct.GetSlots()


def bootstrapped_context(scaling):
    """A loaded GPU context whose bootstrapping setup provides the conjugation key."""
    cc, keys = context(complex_data=True, depth=30, slots=SLOTS, ring=4096, scaling=scaling,
                       bootstrap=True)
    cc.EvalBootstrapSetup([1, 1], [0, 0], SLOTS)
    cc.EvalBootstrapKeyGen(keys.secretKey, SLOTS)
    cc.SetRotationKeyCache(0)  # the conjugation key reloads from host RAM on every use
    cc.LoadContext(keys.publicKey)
    cc.SetCiphertextCache(0)  # the input and the destination are evicted between calls
    return cc, keys


class EvalConjugateTest(unittest.TestCase):
    def test_values_metadata_and_aliases(self):
        for scaling in ("FLEXIBLEAUTO", "FIXEDMANUAL"):
            with self.subTest(scaling=scaling):
                cc, keys = bootstrapped_context(scaling)

                def encrypt(values):
                    return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(values))

                ct = encrypt(X)
                alias = ct
                result = cc.EvalConjugate(ct)
                self.assertIsNot(result, ct)
                check(cc, keys, result, conj(X), TOL)
                self.assertEqual(metadata(result), metadata(ct))
                self.assertIs(alias, ct)
                check(cc, keys, alias, X, TOL)  # the input is untouched
                check(cc, keys, cc.EvalConjugate(result), X, TOL)

                # A product with a pending rescale keeps its degree and scale.
                product = cc.EvalMult(encrypt(X), encrypt(Y))
                if scaling == "FIXEDMANUAL":
                    self.assertEqual(product.GetNoiseScaleDeg(), 2)
                before = metadata(product)
                conjugated = cc.EvalConjugate(product)
                self.assertEqual(metadata(conjugated), before)
                self.assertEqual(metadata(product), before)
                expected = conj([x * y for x, y in zip(X, Y)])
                check(cc, keys, conjugated, expected, TOL)
                # ...and the conjugate keeps working through the rescale and later ops.
                if scaling == "FIXEDMANUAL":
                    cc.RescaleInPlace(conjugated)
                check(cc, keys, cc.EvalMult(conjugated, 2.0), [2 * v for v in expected], TOL)

                # A rescaled, lower-level input.
                low = cc.EvalMult(encrypt(X), 0.5)
                if scaling == "FIXEDMANUAL":
                    cc.RescaleInPlace(low)
                check(cc, keys, cc.EvalConjugate(low), conj([0.5 * v for v in X]), TOL)

                # An offloaded input is reloaded first.
                off = encrypt(X)
                off.Offload()
                check(cc, keys, cc.EvalConjugate(off), conj(X), TOL)

    def test_recovers_packed_channels(self):
        cc, keys = bootstrapped_context("FLEXIBLEAUTO")
        n = cc.GetRingDimension()
        z = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(X))
        z_conj = cc.EvalConjugate(z)
        real = cc.EvalMult(cc.EvalAdd(z, z_conj), 0.5)
        # (z - conj(z)) / (2i): X^(3N/2) multiplies every slot by -i exactly.
        imag = cc.EvalMult(cc.MultByMonomial(cc.EvalSub(z, z_conj), 3 * n // 2), 0.5)
        check(cc, keys, real, [complex(v.real) for v in X], TOL)
        check(cc, keys, imag, [complex(v.imag) for v in X], TOL)

    def test_errors(self):
        cc, keys = bootstrapped_context("FLEXIBLEAUTO")
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(X))

        # Encrypt on the CPU, then select GPUs without loading them.
        other, other_keys = context(devices=[], complex_data=True)
        foreign = other.Encrypt(other_keys.publicKey, other.MakeCKKSPackedPlaintext(X))
        with self.assertRaises(RuntimeError):  # CPU-only context
            other.EvalConjugate(foreign)
        with self.assertRaises(ValueError):
            cc.EvalConjugate(foreign)
        with self.assertRaises(ValueError):
            cc.EvalConjugate(None)

        # GPU context that was never loaded.
        other.SetDevices(DEVICES)
        with self.assertRaisesRegex(RuntimeError, "LoadContext"):
            other.EvalConjugate(foreign)

        # Loaded GPU context without a bootstrapping setup, hence without the conjugation key.
        other.LoadContext(other_keys.publicKey)
        with self.assertRaisesRegex(ValueError, "conjugation key"):
            other.EvalConjugate(foreign)

        # Same context, different secret key: its key set has no conjugation key.
        second = cc.KeyGen()
        with self.assertRaisesRegex(ValueError, "conjugation key"):
            cc.EvalConjugate(cc.Encrypt(second.publicKey, cc.MakeCKKSPackedPlaintext(X)))

        check(cc, keys, cc.EvalConjugate(ct), conj(X), TOL)  # the context still works


if __name__ == "__main__":
    unittest.main()
