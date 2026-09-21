#!/usr/bin/env python3
"""
M2 — multi-chunk SSD scan as a real ttnn op-graph, PCC-checked on ttsim against the naive recurrence golden.

Implements all four SSD terms over C chunks with a host loop carrying the (N,P) state:
  per chunk c (local log-decays a_c, a_cs = cumsum(a_c)):
    L_c[i,j]      = exp(a_cs[i]-a_cs[j])  for i>=j  (intra-chunk decay mask)
    Ydiag_c       = ((C_c·B_cᵀ) ∘ L_c) · xd_c                      [diagonal / intra-chunk]
    Yoff_c        = entry_decay_c ∘ (C_c · S_in)                   [off-diagonal / state read]
    contrib_c     = (decay_states_c ∘ B_c)ᵀ · xd_c                 [chunk final-state build]
    S_in(c+1)     = total_decay_c · S_in(c) + contrib_c           [inter-chunk recurrence]
  where entry_decay_c[q]=exp(a_cs[q]), decay_states_c[q]=exp(a_cs[-1]-a_cs[q]), total_decay_c=exp(a_cs[-1]).

Decay factors are host-computed here (device cumsum/segsum is a later milestone); the matmuls + scaled
adds + the carried state all run in ttnn on the simulator. torch-free. Run in the release Docker + ttsim.
"""
import sys, os, numpy as np
import ttnn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ssd_ref"))
from ssd_minimal import softplus  # noqa: E402

np.random.seed(1)
TILE = ttnn.TILE_LAYOUT
F32 = ttnn.float32
L, Q, N, P = 128, 64, 64, 64      # 2 chunks
NC = L // Q


def tt(d, a):
    return ttnn.to_layout(ttnn.Tensor(np.ascontiguousarray(a.astype(np.float32)), F32).to(d, ttnn.DRAM_MEMORY_CONFIG), TILE)


def rb(t):
    try:
        return np.array(t.cpu().to_numpy(), dtype=np.float32)
    except Exception:
        return np.array(ttnn.typecast(t, F32).cpu().to_numpy(), dtype=np.float32)


def pcc(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return 1.0 if np.allclose(a, b) else float(np.corrcoef(a, b)[0, 1])


def make_inputs():
    C = np.random.randn(L, N); B = np.random.randn(L, N); x = np.random.randn(L, P)
    dt = softplus(np.random.randn(L)); A = -abs(np.random.rand()) - 0.1
    return C, B, x, dt, A


def golden_naive(C, B, x, dt, A):
    xd = x * dt[:, None]; a = np.exp(dt * A)
    h = np.zeros((P, N)); ys = []
    for t in range(L):
        h = a[t] * h + xd[t][:, None] * B[t][None, :]
        ys.append(h @ C[t])
    return np.stack(ys, 0)                       # (L,P)


def decay_factors(a_chunk):
    acs = np.cumsum(a_chunk)                                   # (Q,)
    Lm = np.tril(np.exp(acs[:, None] - acs[None, :]))          # (Q,Q) intra-chunk
    entry = np.exp(acs)                                        # (Q,)
    dstate = np.exp(acs[-1] - acs)                             # (Q,)
    total = np.exp(acs[-1])                                    # scalar
    return Lm, entry, dstate, total


def run_numpy_chunked(C, B, x, dt, A):
    """Host per-chunk loop — sanity that the chunked formulation equals the naive golden."""
    xd = x * dt[:, None]; a = dt * A
    S = np.zeros((N, P)); ys = []
    for c in range(NC):
        s = slice(c * Q, (c + 1) * Q)
        Cc, Bc, xc, ac = C[s], B[s], xd[s], a[s]
        Lm, entry, dstate, total = decay_factors(ac)
        Ydiag = ((Cc @ Bc.T) * Lm) @ xc
        Yoff = entry[:, None] * (Cc @ S)
        ys.append(Ydiag + Yoff)
        S = total * S + (dstate[:, None] * Bc).T @ xc
    return np.concatenate(ys, 0)


def run_ttnn_chunked(d, C, B, x, dt, A):
    xd = x * dt[:, None]; a = dt * A
    S = ttnn.mul(tt(d, np.ones((N, P))), 0.0)                  # zero (N,P) state on device
    ys = []
    for c in range(NC):
        s = slice(c * Q, (c + 1) * Q)
        Cc, Bc, xc, ac = C[s], B[s], xd[s], a[s]
        Lm, entry, dstate, total = decay_factors(ac)
        Cct, Bct, xct, Lt = tt(d, Cc), tt(d, Bc), tt(d, xc), tt(d, Lm)
        entryt = tt(d, entry[:, None]); dstatet = tt(d, dstate[:, None])
        # Ydiag = ((Cc·Bcᵀ)∘L)·xc
        Ydiag = ttnn.matmul(ttnn.mul(ttnn.matmul(Cct, ttnn.transpose(Bct, -2, -1)), Lt), xct)
        # Yoff = entry ∘ (Cc·S)
        Yoff = ttnn.mul(ttnn.matmul(Cct, S), entryt)
        ys.append(rb(ttnn.add(Ydiag, Yoff)))
        # S = total·S + (dstate∘Bc)ᵀ·xc
        contrib = ttnn.matmul(ttnn.transpose(ttnn.mul(Bct, dstatet), -2, -1), xct)
        S = ttnn.add(ttnn.mul(S, float(total)), contrib)
    return np.concatenate(ys, 0)


def main():
    C, B, x, dt, A = make_inputs()
    gold = golden_naive(C, B, x, dt, A)
    npc = run_numpy_chunked(C, B, x, dt, A)
    print(f"  numpy chunked vs naive golden   PCC = {pcc(npc, gold):.6f}   (formulation check)")
    d = ttnn.open_device(device_id=0)
    try:
        tnn = run_ttnn_chunked(d, C, B, x, dt, A)
    finally:
        ttnn.close_device(d)
    m2 = pcc(tnn, gold)
    print(f"  M2 ttnn multi-chunk SSD scan    PCC = {m2:.6f}   {'PASS' if m2 >= 0.99 else 'FAIL'}")


if __name__ == "__main__":
    main()
