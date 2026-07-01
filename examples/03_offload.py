#!/usr/bin/env python3
"""Ciphertext offload/reload and reclaiming VRAM.

Useful when you hold more ciphertexts than fit in VRAM at once, or want to hand some
GPU memory back before other work: offload the ones not currently needed, reload
(implicitly or explicitly) right before they're used again. Offload/reload is a
bit-exact round trip -- no decrypt, rescale, or NTT/domain change -- so it never
changes what the ciphertext represents. cc.TrimGPUMemoryPool() returns the freed
memory to the OS (Offload() alone only pools it for reuse).

Runs in seconds with <1 GB of GPU memory (toy, NON-SECURE parameters).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find fideslib_py
import fideslib_py as fhe

# ---------------------------------------------------------------- parameters
params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_NotSet)
params.SetRingDim(1 << 13)
params.SetMultiplicativeDepth(4)
params.SetScalingModSize(50)
params.SetFirstModSize(60)
params.SetNumLargeDigits(3)
params.SetBatchSize(1 << 12)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)
params.SetDevices([0])

cc = fhe.GenCryptoContext(params)
cc.Enable(fhe.PKE)
cc.Enable(fhe.KEYSWITCH)
cc.Enable(fhe.LEVELEDSHE)

keys = cc.KeyGen()
cc.LoadContext(keys.publicKey)

x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
pt_x = cc.MakeCKKSPackedPlaintext(x)
ct = cc.Encrypt(keys.publicKey, pt_x)


def dec(c, n=len(x)):
    p = cc.Decrypt(keys.secretKey, c)
    p.SetLength(n)
    return p.GetRealPackedValue()


# ---------------------------------------------------------------- offload / reload
assert not ct.IsOffloaded(), "freshly-encrypted ciphertext should be GPU-resident"

ct.Offload()
assert ct.IsOffloaded(), "Offload() should mark the ciphertext as offloaded"
print("Offloaded.")

# Bit-exact round trip: reload and decrypt should reproduce the original values.
ct.Reload()
assert not ct.IsOffloaded(), "Reload() should clear the offloaded flag"
values = dec(ct)
print(f"After offload+reload: {[round(v, 6) for v in values]}")
assert all(abs(a - b) < 1e-6 for a, b in zip(values, x))

# Offload again and use it directly: every op transparently reloads on first use,
# so calling Reload() first is optional.
ct.Offload()
assert ct.IsOffloaded()
ct_twice = cc.EvalAdd(ct, ct)  # implicit reload of `ct` inside EvalAdd
assert not ct.IsOffloaded(), "EvalAdd should have transparently reloaded ct"

values = dec(ct_twice)
print(f"(x+x) after implicit reload: {[round(v, 6) for v in values]}")
assert all(abs(a - 2 * b) < 1e-6 for a, b in zip(values, x))

# Decrypt also transparently reloads.
ct.Offload()
assert ct.IsOffloaded()
values = dec(ct)
assert not ct.IsOffloaded(), "Decrypt should have transparently reloaded ct"
assert all(abs(a - b) < 1e-6 for a, b in zip(values, x))

# Idempotence: double offload / double reload are no-ops.
ct.Offload()
ct.Offload()
assert ct.IsOffloaded()
ct.Reload()
ct.Reload()
assert not ct.IsOffloaded()

# ---------------------------------------------------------------- reclaim VRAM
# Offload() frees the limbs into FIDESlib's internal GPU memory pool for cheap reuse,
# but does NOT return that memory to the OS (nvidia-smi will not show a drop). When you
# actually want the VRAM back -- e.g. before other, non-FIDESlib GPU work -- offload the
# ciphertexts you want to keep and then trim the pool. (No trim is needed if you only
# intend to reuse the memory for more FIDESlib ciphertexts/ops; the pool recycles it.)
cached = [cc.Encrypt(keys.publicKey, pt_x) for _ in range(8)]
for c in cached:
    c.Offload()
cc.TrimGPUMemoryPool()  # freed VRAM handed back to the system
print("Offloaded a batch and trimmed the GPU memory pool.")

# The ciphertexts are still usable afterwards -- they reload on first use.
values = dec(cc.EvalAdd(cached[0], cached[1]))
assert all(abs(a - 2 * b) < 1e-6 for a, b in zip(values, x))

print("\nAll checks passed.")
