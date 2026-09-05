"""One multi-GPU case, run in its own process.

Kept as a standalone script rather than a pytest test because the failures under
investigation abort the process (F3) or corrupt it at teardown (F5): only a subprocess
boundary lets the harness tell "wrong value" from "died". test_multigpu.py drives it.

Usage: mgpu_case.py '<json config>'
Config keys: devices, dnum, depth, logN, sizes, op ("accumulate"|"rotate"|"chain")
Prints one JSON object per line to stdout; exit code 0 only if every check passed.
"""

import gc
import json
import sys

sys.path.insert(0, "/home/falcettaa/PyFIDESlib")
import fideslib_py as fhe  # noqa: E402


def build(cfg):
    p = fhe.CCParams()
    p.SetSecurityLevel(fhe.HEStd_NotSet)
    p.SetRingDim(1 << cfg["logN"])
    p.SetMultiplicativeDepth(cfg["depth"])
    p.SetScalingModSize(50)
    p.SetFirstModSize(60)
    p.SetNumLargeDigits(cfg["dnum"])
    p.SetBatchSize(1 << (cfg["logN"] - 1))
    p.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    p.SetKeySwitchTechnique(fhe.HYBRID)
    p.SetDevices(cfg["devices"])

    cc = fhe.GenCryptoContext(p)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)
    return cc


def emit(name, ok, **extra):
    print(json.dumps({"check": name, "ok": bool(ok), **extra}), flush=True)
    return bool(ok)


def close(got, exp, tol=1e-4):
    # NaN compares false against everything, which is exactly the F4 signature.
    return abs(got - exp) < tol


def run_accumulate(cc, keys, cfg):
    """F3/F4 path: Ciphertext::rotate_hoisted(..., ext=true) via AccumulateSum."""
    sizes = cfg["sizes"]
    idxs = set()
    for n in sizes:
        idxs |= set(fhe.accumulate_rotation_indices(n, stride=1))
    cc.EvalRotateKeyGen(keys.secretKey, sorted(idxs))
    cc.LoadContext(keys.publicKey)

    x = [float(i + 1) for i in range(max(sizes))]
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
    ok = True
    for n in sizes:
        r = cc.AccumulateSum(ct, n)
        pt = cc.Decrypt(keys.secretKey, r)
        pt.SetLength(1)
        got, exp = pt.GetRealPackedValue()[0], sum(x[:n])
        ok &= emit(f"accumulate/{n}", close(got, exp), got=got, expected=exp)
    return ok


def run_rotate(cc, keys, cfg):
    """Plain (non-extended) hoisted rotation -- the path bootstrap uses."""
    steps = [1, 2, 4, 8]
    cc.EvalRotateKeyGen(keys.secretKey, steps)
    cc.LoadContext(keys.publicKey)

    n = 64
    x = [float(i + 1) for i in range(n)]
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
    ok = True
    for s in steps:
        r = cc.EvalRotate(ct, s)
        pt = cc.Decrypt(keys.secretKey, r)
        pt.SetLength(n)
        got, exp = pt.GetRealPackedValue()[0], x[s]
        ok &= emit(f"rotate/{s}", close(got, exp), got=got, expected=exp)
    return ok


def run_chain(cc, keys, cfg):
    """Multiplicative chain with rescale + interleaved rotation: the key-switch path
    that already works, kept as a control so a regression here is not mistaken for F4."""
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, [1])
    cc.LoadContext(keys.publicKey)

    n = 8
    x = [float(i + 1) for i in range(n)]
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
    acc, exp = ct, list(x)
    depth = min(cfg["depth"] - 1, 6)
    for _ in range(depth):
        acc = cc.EvalMult(acc, ct)
        exp = [a * b for a, b in zip(exp, x)]
    acc = cc.EvalRotate(acc, 1)
    exp = exp[1:] + [0.0]
    pt = cc.Decrypt(keys.secretKey, acc)
    pt.SetLength(1)
    got = pt.GetRealPackedValue()[0]
    return emit(f"chain/{depth}", close(got, exp[0], tol=1e-2 * max(1.0, abs(exp[0]))),
                got=got, expected=exp[0])


OPS = {"accumulate": run_accumulate, "rotate": run_rotate, "chain": run_chain}


def main():
    cfg = json.loads(sys.argv[1])
    cfg.setdefault("logN", 14)
    cfg.setdefault("sizes", [2, 4, 8, 64])
    cfg.setdefault("op", "accumulate")

    cc = build(cfg)
    keys = cc.KeyGen()
    ok = OPS[cfg["op"]](cc, keys, cfg)

    # F5 aborts inside ~ContextData. Drop the context here, while we can still report it,
    # rather than at interpreter shutdown where the abort has no line to attach to.
    del keys, cc
    gc.collect()
    ok &= emit("teardown", True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
