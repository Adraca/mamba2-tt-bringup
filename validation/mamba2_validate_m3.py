#!/usr/bin/env python3
"""
M3 — full Mamba-2 mixer block validated on ttsim against the numpy golden (mamba2_block.py).

ttnn (on-device) ops this milestone: in_proj matmul, the per-head chunked SSD scan (M2), the gated RMSNorm
(y*silu(z) -> rms_norm -> *w), and out_proj matmul. Host glue this milestone (clearly labeled, all cheap /
layout-awkward): the in_proj output split, softplus(dt+bias), the depthwise causal conv1d + its silu, and the
x/B/C split. Moving conv1d + softplus on-device is the next milestone (M3b) toward the fused model-dir graph.

PCC target 0.99 vs the full numpy block. torch-free. Run in the release Docker + ttsim.
"""
import sys, os, numpy as np
import ttnn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ssd_ref"))
from mamba2_block import default_config, make_weights, mamba2_block, silu, causal_conv1d  # noqa: E402
from ssd_minimal import softplus  # noqa: E402
from mamba2_validate_m2 import tt, rb, pcc  # reuse the validated ttnn tensor helpers  # noqa: E402

TILE = ttnn.TILE_LAYOUT
F32 = ttnn.float32
EPS = 1e-5


def ssd_chunked_ttnn(d, C, B, x, dt, A, chunk=64):
    """Dimension-agnostic per-head chunked SSD in ttnn (same math as M2, dims derived from inputs).
    C,B:(L,N)  x:(L,P)  dt:(L,)  A:scalar -> y:(L,P) numpy. Decay factors host-computed.
    Returns the PURE scan WITHOUT the D skip — the caller adds `D * x` (matching ssd_naive, which applies
    it internally). `entry[q]=exp(acs[q])` carries the chunk-entry state forward to position q (no double
    decay): this is the exact formula M2 validated at PCC 1.000000 vs the naive recurrence."""
    L, N = B.shape
    P = x.shape[1]
    nc = L // chunk
    xd = x * dt[:, None]; a = dt * A
    S = ttnn.mul(tt(d, np.zeros((N, P))), 1.0)                       # (N,P) zero state on device
    ys = []
    for c in range(nc):
        s = slice(c * chunk, (c + 1) * chunk)
        Cc, Bc, xc, ac = C[s], B[s], xd[s], a[s]
        acs = np.cumsum(ac)
        Lm = np.tril(np.exp(acs[:, None] - acs[None, :]))
        entry = np.exp(acs)[:, None]; dstate = np.exp(acs[-1] - acs)[:, None]; total = float(np.exp(acs[-1]))
        Cct, Bct, xct, Lt = tt(d, Cc), tt(d, Bc), tt(d, xc), tt(d, Lm)
        Ydiag = ttnn.matmul(ttnn.mul(ttnn.matmul(Cct, ttnn.transpose(Bct, -2, -1)), Lt), xct)
        Yoff = ttnn.mul(ttnn.matmul(Cct, S), tt(d, entry))
        ys.append(rb(ttnn.add(Ydiag, Yoff)))
        contrib = ttnn.matmul(ttnn.transpose(ttnn.mul(Bct, tt(d, dstate)), -2, -1), xct)
        S = ttnn.add(ttnn.mul(S, total), contrib)
    return np.concatenate(ys, 0)


def ttnn_rmsnorm_gated(d, y, z, norm_w):
    """g = rmsnorm(y * silu(z)) * w, all in ttnn. y,z:(L,di) np; norm_w:(di,) np -> (L,di) np."""
    yt, zt, wt = tt(d, y), tt(d, z), tt(d, norm_w[None, :])
    g = ttnn.mul(yt, ttnn.silu(zt))                                  # gate
    try:
        normed = ttnn.rms_norm(g, epsilon=EPS, weight=tt(d, norm_w))
        return rb(normed)
    except Exception:
        sq = ttnn.mul(g, g)
        ms = ttnn.mean(sq, dim=-1, keepdim=True)                     # (L,1)
        inv = ttnn.rsqrt(ttnn.add(ms, EPS))
        return rb(ttnn.mul(ttnn.mul(g, inv), wt))


def run_ttnn_block(u, W, cfg):
    di, ng, ds, nh = W["d_inner"], W["ng"], W["ds"], W["nheads"]
    hd = cfg["headdim"]
    d = ttnn.open_device(device_id=0)
    try:
        # in_proj (ttnn matmul), then split on host
        ut = tt(d, u)
        zxbcdt = rb(ttnn.matmul(ut, ttnn.transpose(tt(d, W["in_proj"]), -2, -1)))   # (L, proj_in)
        z = zxbcdt[:, :di]
        xBC = zxbcdt[:, di:di + W["conv_dim"]]
        dt = zxbcdt[:, di + W["conv_dim"]:di + W["conv_dim"] + nh]
        # host: softplus(dt), causal conv1d + silu, x/B/C split
        dt = softplus(dt + W["dt_bias"][None, :])
        xBC = silu(causal_conv1d(xBC, W["conv_w"], W["conv_b"]))
        x = xBC[:, :di]; B = xBC[:, di:di + ng * ds]; C = xBC[:, di + ng * ds:]
        A = -np.exp(W["A_log"])
        Bg = B.reshape(-1, ng, ds)[:, 0, :]; Cg = C.reshape(-1, ng, ds)[:, 0, :]
        # per-head chunked SSD in ttnn (reuse M2), then D skip
        x_h = x.reshape(-1, nh, hd)
        ys = []
        for h in range(nh):
            yh = ssd_chunked_ttnn(d, Cg, Bg, x_h[:, h, :], dt[:, h], float(A[h]), chunk=cfg["chunk"])
            yh = yh + W["D"][h] * x_h[:, h, :]                                        # D skip
            ys.append(yh)
        y = np.concatenate(ys, axis=1)                                               # (L, d_inner)
        # gated RMSNorm (ttnn), then out_proj (ttnn matmul)
        g = ttnn_rmsnorm_gated(d, y, z, W["norm_w"])
        out = rb(ttnn.matmul(tt(d, g), ttnn.transpose(tt(d, W["out_proj"]), -2, -1)))
        return out
    finally:
        ttnn.close_device(d)


def main():
    cfg = default_config()
    W = make_weights(cfg, seed=0)
    rng = np.random.default_rng(7)
    u = rng.standard_normal((cfg["seqlen"], cfg["d_model"]))
    gold = mamba2_block(u, W, cfg)
    out = run_ttnn_block(u, W, cfg)
    p = pcc(out, gold)
    print(f"  M3 Mamba2 block, tensor ops on ttnn   PCC = {p:.6f}   {'PASS' if p >= 0.99 else 'FAIL'}")
    print(f"     ON-DEVICE (validated): in_proj + per-head chunked SSD + gated RMSNorm + out_proj")
    print(f"     HOST this milestone (M3b moves on-device): causal conv1d, softplus(dt), the projection splits")


if __name__ == "__main__":
    main()
