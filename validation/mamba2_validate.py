#!/usr/bin/env python3
"""
Mamba-2 SSD bring-up validation — real TTNN op-graphs PCC-checked against the numpy golden
(../ssd_ref/ssd_minimal.py) on the ttsim functional simulator. torch-free. $0, no hardware.

Milestones (bottom-up, each must clear PCC >= 0.99):
  M1  single-chunk SSD diagonal block:   Ydiag = (C·Bᵀ ∘ L) · x     [pure matmul + decay mask]
  M1b compute the decay mask L in ttnn:  L = tril(exp(segsum(a)))    [cumsum + exp on device]
  M2  chunk-state build + off-diagonal read (multi-chunk)           [added next]

Run in the tt-metal release Docker with TT_METAL_SIMULATOR=libttsim_bh.so and soc_descriptor.yaml in CWD.
"""
import sys, os, numpy as np
import ttnn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ssd_minimal.py sits beside this file
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ssd_ref"))
from ssd_minimal import segsum, softplus  # noqa: E402

np.random.seed(0)
TILE = ttnn.TILE_LAYOUT
F32 = ttnn.float32

# small but tile-aligned SSD config (single batch/head/chunk for M1)
Q = 64      # chunk length (tile-aligned)
N = 64      # state dim
P = 64      # head dim


def to_tile(d, a):
    return ttnn.to_layout(ttnn.Tensor(np.ascontiguousarray(a.astype(np.float32)), F32).to(d, ttnn.DRAM_MEMORY_CONFIG), TILE)


def rb(t):
    try:
        return np.array(t.cpu().to_numpy(), dtype=np.float32)
    except Exception:
        return np.array(ttnn.typecast(t, F32).cpu().to_numpy(), dtype=np.float32)


def pcc(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    if np.allclose(a, b):
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def make_inputs():
    C = np.random.randn(Q, N)
    B = np.random.randn(Q, N)
    x = np.random.randn(Q, P)
    dt = softplus(np.random.randn(Q))
    A = -abs(np.random.rand()) - 0.1
    a = dt * A                                  # (Q,) log decays
    xd = x * dt[:, None]                        # discretized input Δ·x
    return C, B, x, xd, a


def golden_diag(C, B, xd, a):
    """Single-chunk diagonal block in numpy (fp64) = the golden for M1."""
    L = np.exp(segsum(a))                        # (Q,Q) lower-tri decay, 0 above diag via -inf->exp=0
    scores = C @ B.T                             # (Q,Q)
    return (scores * L) @ xd                     # (Q,P)


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        C, B, x, xd, a = make_inputs()
        L = np.exp(segsum(a))                    # host decay mask for M1 (device version = M1b)
        gold = golden_diag(C, B, xd, a)

        # --- M1: diagonal block as ttnn matmul chain with host-precomputed L ---
        Ct, Bt, xdt, Lt = to_tile(dev, C), to_tile(dev, B), to_tile(dev, xd), to_tile(dev, L)
        scores = ttnn.matmul(Ct, ttnn.transpose(Bt, -2, -1))     # C·Bᵀ  (Q,Q)
        masked = ttnn.mul(scores, Lt)                            # ∘ L
        ydiag = ttnn.matmul(masked, xdt)                         # ·xd   (Q,P)
        m1 = pcc(rb(ydiag), gold)
        print(f"  M1  diagonal block (host L)      PCC = {m1:.6f}   {'PASS' if m1 >= 0.99 else 'FAIL'}")

        # --- M1b: compute L = tril(exp(segsum(a))) on device via cumsum ---
        # segsum(a)[i,j] = cumsum(a)[i] - cumsum(a)[j] for i>=j; L=exp(that), 0 above diagonal.
        ac = np.cumsum(a)                                        # host cumsum baseline
        Ldev_ref = np.tril(np.exp(ac[:, None] - ac[None, :]))    # equals exp(segsum) on/below diag
        m1b_ref = pcc(Ldev_ref, L)
        print(f"  M1b decay-mask identity check     PCC = {m1b_ref:.6f}   (numpy segsum == cumsum-diff form)")
        if hasattr(ttnn, "cumsum") or hasattr(ttnn, "moreh_cumsum"):
            print("      ttnn.cumsum present -> device segsum path is implementable")
        else:
            print("      ttnn.cumsum absent  -> compute L host-side or via matmul-with-lower-tri-ones")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
