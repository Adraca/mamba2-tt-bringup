"""
mamba2_block.py — clean-room NUMPY reference (GOLDEN) for the full Mamba-2 mixer block.

Faithful to HF `Mamba2Mixer` (no cache / full-sequence path):
  zxbcdt = in_proj(u)
  z, xBC, dt = split(zxbcdt, [d_inner, d_inner + 2*ngroups*d_state, nheads])
  dt = softplus(dt + dt_bias)
  xBC = silu(causal_depthwise_conv1d(xBC))
  x, B, C = split(xBC, [d_inner, ngroups*d_state, ngroups*d_state])
  y = SSD(x_heads, dt, A=-exp(A_log), B, C, D)          # the scan (uses ssd_naive here = independent oracle)
  y = rmsnorm(y * silu(z)) * norm_w                      # gated RMSNorm
  out = out_proj(y)

Small but faithful config (real ratios). torch-free. Weights are supplied so the ttnn port validates against
exactly this. See ../design/SSD_MATH_AND_TT_MAPPING.md and README.
"""
import numpy as np
from ssd_minimal import ssd_naive, softplus


def silu(x):
    return x / (1.0 + np.exp(-x))


def rmsnorm(x, w, eps=1e-5):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def causal_conv1d(x, weight, bias):
    """x:(L,Cc) depthwise causal conv, weight:(Cc,K), bias:(Cc,) -> (L,Cc). Left zero-pad by K-1."""
    L, Cc = x.shape
    K = weight.shape[1]
    xp = np.concatenate([np.zeros((K - 1, Cc)), x], axis=0)          # causal left pad
    out = np.zeros((L, Cc))
    for k in range(K):
        out += xp[k:k + L] * weight[:, k][None, :]
    return out + bias[None, :]


def default_config():
    return dict(d_model=64, expand=2, headdim=32, ngroups=1, d_state=32, conv_k=4, seqlen=128, chunk=64)


def make_weights(cfg, seed=0):
    rng = np.random.default_rng(seed)
    dm, e, hd, ng, ds = cfg["d_model"], cfg["expand"], cfg["headdim"], cfg["ngroups"], cfg["d_state"]
    d_inner = e * dm
    nheads = d_inner // hd
    conv_dim = d_inner + 2 * ng * ds
    proj_in = 2 * d_inner + 2 * ng * ds + nheads
    s = 0.1
    return dict(
        nheads=nheads, d_inner=d_inner, conv_dim=conv_dim, ng=ng, ds=ds,
        in_proj=rng.standard_normal((proj_in, dm)) * s,
        conv_w=rng.standard_normal((conv_dim, cfg["conv_k"])) * s,
        conv_b=rng.standard_normal(conv_dim) * s,
        dt_bias=rng.standard_normal(nheads) * s,
        A_log=np.log(rng.uniform(1.0, 16.0, nheads)),                # HF-style init: A = -exp(A_log) in [-16,-1]
        D=rng.standard_normal(nheads) * s,
        norm_w=np.ones(d_inner) + rng.standard_normal(d_inner) * s,
        out_proj=rng.standard_normal((dm, d_inner)) * s,
    )


def mamba2_block(u, W, cfg):
    """u:(L,d_model) -> out:(L,d_model). Reference forward using the naive scan oracle."""
    L = u.shape[0]
    di, ng, ds, nh = W["d_inner"], W["ng"], W["ds"], W["nheads"]
    hd = cfg["headdim"]
    zxbcdt = u @ W["in_proj"].T                                      # (L, proj_in)
    z = zxbcdt[:, :di]
    xBC = zxbcdt[:, di:di + W["conv_dim"]]
    dt = zxbcdt[:, di + W["conv_dim"]:]                              # (L, nheads)
    dt = softplus(dt + W["dt_bias"][None, :])
    xBC = silu(causal_conv1d(xBC, W["conv_w"], W["conv_b"]))
    x = xBC[:, :di]
    B = xBC[:, di:di + ng * ds]                                      # (L, ng*ds); ng=1 -> shared
    C = xBC[:, di + ng * ds:]
    A = -np.exp(W["A_log"])                                          # (nheads,)
    assert ng == 1, "this reference implements the ngroups=1 case (B,C shared across heads)"
    # per-head scan (ngroups=1: B,C shared across heads). x_h:(L,headdim), reuse ssd_naive per head.
    # NOTE: ssd_naive applies the D skip internally (y + D*x), so the per-head y already includes it.
    x_h = x.reshape(L, nh, hd)
    Bg = B.reshape(L, ng, ds)[:, 0, :]                              # (L, ds)
    Cg = C.reshape(L, ng, ds)[:, 0, :]
    ys = []
    for h in range(nh):
        xh = x_h[:, h, :][None]                                     # (1,L,headdim)  -> ssd_naive wants (B,L,H,P)
        yh = ssd_naive(xh[..., None, :], dt[None, :, h:h + 1],
                       A[h:h + 1], Bg[None, :, None, :], Cg[None, :, None, :], W["D"][h:h + 1])
        ys.append(yh[0, :, 0, :])                                   # (L, headdim)
    y = np.concatenate(ys, axis=1)                                  # (L, d_inner)
    y = rmsnorm(y * silu(z), W["norm_w"])                           # gated RMSNorm
    return y @ W["out_proj"].T                                      # (L, d_model)


if __name__ == "__main__":
    cfg = default_config()
    W = make_weights(cfg)
    rng = np.random.default_rng(7)
    u = rng.standard_normal((cfg["seqlen"], cfg["d_model"]))
    out = mamba2_block(u, W, cfg)
    print(f"mamba2_block: in {u.shape} -> out {out.shape}; "
          f"finite={np.isfinite(out).all()} mean={out.mean():.4f} std={out.std():.4f}")
