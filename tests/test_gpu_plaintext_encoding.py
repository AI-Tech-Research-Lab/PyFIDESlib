"""GPU plaintext encoding (CryptoContext.SetGPUPlaintextEncoding, on by default).

    CUDA_VISIBLE_DEVICES=3 python3 tests/test_gpu_plaintext_encoding.py
    CUDA_VISIBLE_DEVICES=2,3 PYFIDES_TEST_DEVICES=0,1 python3 tests/test_gpu_plaintext_encoding.py

A plaintext encoded on the GPU must behave bit for bit like one encoded on the CPU: the same
ciphertext multiplied or added with either gives exactly the same ciphertext, also after the
plaintext cache evicts and reloads it. Ciphertexts are compared through their binary
serialization, since OpenFHE's CKKS decoding adds fresh noise on every decryption. FIDESlib's
GpuPlaintextEncodingTests compare the device limbs directly. Parameters are small, non-secure
test parameters.
"""

import os
import random
import tempfile
import unittest

try:  # only for the MakeCKKSPackedPlaintextF64 case
    import numpy as np
except ImportError:
    np = None

from test_upstream_220 import DEVICES, context, decode, fhe

SLOTS = 64


def values(n, seed, complex_data=False, scale=1.0):
    rng = random.Random(seed)
    if complex_data:
        return [complex(rng.uniform(-scale, scale), rng.uniform(-scale, scale)) for _ in range(n)]
    return [rng.uniform(-scale, scale) for _ in range(n)]


def serialized(ct):
    """The ciphertext's exact bits (decryption is randomized, so it cannot compare bits)."""
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    try:
        fhe.SerializeToFile(path, ct, fhe.BINARY)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


def pair(cc, make):
    """The same plaintext encoded on the CPU and on the GPU."""
    cc.SetGPUPlaintextEncoding(False)
    cpu = make()
    cc.SetGPUPlaintextEncoding(True)
    gpu = make()
    return cpu, gpu


def loaded_context(**kwargs):
    cc, keys = context(slots=SLOTS, **kwargs)
    cc.LoadContext(keys.publicKey)
    return cc, keys


class GpuPlaintextEncodingTest(unittest.TestCase):
    def test_default_and_toggle(self):
        cc, _ = loaded_context()
        self.assertTrue(cc.GetGPUPlaintextEncoding())
        self.assertTrue(cc.MakeCKKSPackedPlaintext([1.0, 2.0]).IsGPUEncoded())
        cc.SetGPUPlaintextEncoding(False)
        self.assertFalse(cc.GetGPUPlaintextEncoding())
        self.assertFalse(cc.MakeCKKSPackedPlaintext([1.0, 2.0]).IsGPUEncoded())

    def test_cpu_only_context_encodes_on_cpu(self):
        cc, keys = context(devices=[], slots=SLOTS)
        pt = cc.MakeCKKSPackedPlaintext([0.5, 0.25])
        self.assertFalse(pt.IsGPUEncoded())
        ct = cc.Encrypt(keys.publicKey, pt)
        self.assertAlmostEqual(decode(cc, keys, ct, 2)[0].real, 0.5, places=6)

    def test_operations_match_cpu_encoding_bit_for_bit(self):
        for scaling in ("FLEXIBLEAUTO", "FLEXIBLEAUTOEXT", "FIXEDMANUAL", "FIXEDAUTO"):
            for complex_data in (False, True):
                with self.subTest(scaling=scaling, complex_data=complex_data):
                    cc, keys = loaded_context(scaling=scaling, complex_data=complex_data)
                    x = values(SLOTS, 1, complex_data)
                    w = values(SLOTS, 2, complex_data)
                    for level in (0, 1, 3):
                        for slots in (SLOTS, 16):
                            with self.subTest(level=level, slots=slots):
                                ct = cc.Encrypt(keys.publicKey,
                                                cc.MakeCKKSPackedPlaintext(x[:slots], 1, level, slots))
                                cpu, gpu = pair(cc, lambda: cc.MakeCKKSPackedPlaintext(w[:slots], 1, level, slots))
                                self.assertFalse(cpu.IsGPUEncoded())
                                self.assertTrue(gpu.IsGPUEncoded())
                                self.assertEqual(cpu.GetLevel(), gpu.GetLevel())
                                for op in ("EvalMult", "EvalAdd", "EvalSub"):
                                    expected = serialized(getattr(cc, op)(ct, cpu))
                                    self.assertEqual(serialized(getattr(cc, op)(ct, gpu)), expected, op)
                                product = decode(cc, keys, cc.EvalMult(ct, gpu), slots)
                                error = max(abs(p - a * b) for p, a, b in zip(product, x, w))
                                self.assertLess(error, 1e-5)

    @unittest.skipIf(np is None, "numpy is needed for MakeCKKSPackedPlaintextF64")
    def test_float64_fast_path_and_wide_values(self):
        cc, keys = loaded_context(scaling="FIXEDMANUAL", depth=4)
        x = np.asarray(values(SLOTS, 3))
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintextF64(x))
        # 300 * 2^50 still fits in a word, 3e6 * 2^50 does not: the wide-value factor path.
        for scale in (1.0, 300.0, 3e6):
            with self.subTest(scale=scale):
                w = np.asarray(values(SLOTS, 4, scale=scale))
                cpu, gpu = pair(cc, lambda: cc.MakeCKKSPackedPlaintextF64(w))
                self.assertTrue(gpu.IsGPUEncoded())
                self.assertEqual(serialized(cc.EvalMult(ct, cpu)), serialized(cc.EvalMult(ct, gpu)))

    def test_noise_scale_degree_two(self):
        cc, keys = loaded_context(scaling="FIXEDMANUAL")
        ct = cc.EvalMult(cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(values(SLOTS, 5))), 1.0)
        self.assertEqual(ct.GetNoiseScaleDeg(), 2)
        level = ct.GetLevel()
        cpu, gpu = pair(cc, lambda: cc.MakeCKKSPackedPlaintext(values(SLOTS, 6), 2, level, SLOTS))
        self.assertEqual(serialized(cc.EvalAdd(ct, cpu)), serialized(cc.EvalAdd(ct, gpu)))

    def test_reload_after_eviction(self):
        cc, keys = loaded_context()
        cc.SetPlaintextCache(0)
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(values(SLOTS, 7)))
        cpu, gpu = pair(cc, lambda: cc.MakeCKKSPackedPlaintext(values(SLOTS, 8)))
        expected = serialized(cc.EvalMult(ct, cpu))
        for _ in range(3):
            self.assertEqual(serialized(cc.EvalMult(ct, gpu)), expected)
            cc.OffloadPlaintexts()
            self.assertFalse(gpu.IsLoadedOnDevice())
            self.assertEqual(cc.GetPlaintextCacheResidentBytes(), 0)
        gpu.UnloadFromDevice()
        self.assertEqual(serialized(cc.EvalMult(ct, gpu)), expected)

    def test_encrypt_and_accessors_of_gpu_encoded_plaintext(self):
        cc, keys = loaded_context(complex_data=True)
        x = values(8, 9, complex_data=True)
        pt = cc.MakeCKKSPackedPlaintext(x)
        self.assertTrue(pt.IsGPUEncoded())
        self.assertEqual(len(pt.GetCKKSPackedValue()), len(x))
        ct = cc.Encrypt(keys.publicKey, pt)  # encodes on the CPU on demand
        error = max(abs(a - b) for a, b in zip(decode(cc, keys, ct, len(x)), x))
        self.assertLess(error, 1e-6)
        self.assertIn("(", str(pt))

    def test_errors_match_cpu_encoding(self):
        cc, _ = loaded_context()
        for gpu in (False, True):
            cc.SetGPUPlaintextEncoding(gpu)
            with self.subTest(gpu=gpu):
                with self.assertRaises(Exception):
                    cc.MakeCKKSPackedPlaintext([1.0] * 8, 1, 0, 4)
                with self.assertRaises(Exception):
                    cc.MakeCKKSPackedPlaintext([1.0] * 4, 1, 100)
                with self.assertRaises(Exception):
                    cc.MakeCKKSPackedPlaintext([1e-20] * 4)


if __name__ == "__main__":
    print(f"devices={DEVICES}")
    unittest.main()
