"""
ssd_minimal.py — clean-room NUMPY reference for the Mamba-2 chunked SSD scan (the GOLDEN).

Implements the chunked State-Space-Duality algorithm from Dao & Gu, "Transformers are SSMs" (Mamba-2),
following the four-term decomposition in ../design/SSD_MATH_AND_TT_MAPPING.md. This is the fp64 numerical
oracle the ttnn implementation is validated against (PCC >= 0.99 on ttsim). Kept torch-free (numpy only) to
match the fleet's ttsim harness pattern and run anywhere. `ssd_naive` is the plain O(L) recurrence used to
prove the chunked result is correct.

Shapes: B=batch, L=seqlen, H=heads, P=head_dim, N=state_dim, Q=chunk_len (L % Q == 0).
"""
import numpy as np


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def segsum(x):
    """Stable segment-sum: L[..., i, j] = sum_{k=j+1..i} x_k for i>=j (else -inf). x: (..., T)."""
    T = x.shape[-1]
    xe = np.broadcast_to(x[..., None], (*x.shape, T)).copy()           # (..., T, T)
    lower = np.tril(np.ones((T, T), dtype=bool), k=-1)
    xe = np.where(lower, xe, 0.0)
    xc = np.cumsum(xe, axis=-2)
    diag = np.tril(np.ones((T, T), dtype=bool), k=0)
    return np.where(diag, xc, -np.inf)


def ssd_chunked(x, dt, A, B, C, D=None, chunk=64):
    """x:(B,L,H,P) dt:(B,L,H) A:(H,) B:(B,L,H,N) C:(B,L,H,N) D:(H,) -> y:(B,L,H,P)."""
    Bb, L, H, P = x.shape
    N = B.shape[-1]
    assert L % chunk == 0, "seqlen must be a multiple of chunk"
    x = x * dt[..., None]                                              # discretize (Δ·x)
    a = dt * A                                                         # (B,L,H) log a_t
    nc = L // chunk
    def ch(t): return t.reshape(Bb, nc, chunk, *t.shape[2:])
    x, a, B, C = ch(x), ch(a), ch(B), ch(C)
    a = np.transpose(a, (0, 3, 1, 2))                                 # (B,H,C,Q)
    a_cumsum = np.cumsum(a, axis=-1)

    # 1. diagonal (intra-chunk) blocks
    Lmat = np.exp(segsum(a))                                          # (B,H,C,Q,Q)
    scores = np.einsum("bcqhn,bckhn->bhcqk", C, B)                    # C Bᵀ per chunk
    Ydiag = np.einsum("bhcqk,bhcqk,bckhp->bcqhp", scores, Lmat, x)

    # 2. chunk final states
    decay_states = np.exp(a_cumsum[..., -1:] - a_cumsum)              # (B,H,C,Q)
    states = np.einsum("bhcq,bcqhn,bcqhp->bchpn", decay_states, B, x)

    # 3. inter-chunk recurrence over chunk states (prepend zero initial state)
    init = np.zeros_like(states[:, :1])
    states = np.concatenate([init, states], axis=1)                  # (B,C+1,H,P,N)
    apad = np.pad(a_cumsum[..., -1], ((0, 0), (0, 0), (1, 0)))        # (B,H,C+1)
    chunk_decay = np.exp(segsum(apad))                               # (B,H,C+1,C+1)
    new_states = np.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)
    states = new_states[:, :-1]                                      # carried-in state per chunk

    # 4. off-diagonal (state -> output) blocks
    state_decay = np.exp(a_cumsum)                                    # (B,H,C,Q)
    Yoff = np.einsum("bcqhn,bchpn,bhcq->bcqhp", C, states, state_decay)

    y = (Ydiag + Yoff).reshape(Bb, L, H, P)
    if D is not None:
        xu = (x.reshape(Bb, L, H, P) / dt[..., None])                # undiscretized x
        y = y + xu * D[None, None, :, None]
    return y


def ssd_naive(x, dt, A, B, C, D=None):
    """Plain O(L) recurrence — the correctness oracle for ssd_chunked."""
    Bb, L, H, P = x.shape
    N = B.shape[-1]
    xd = x * dt[..., None]
    a = np.exp(dt * A)                                                # (B,L,H)
    h = np.zeros((Bb, H, P, N))
    ys = []
    for t in range(L):
        h = a[:, t, :, None, None] * h + xd[:, t, :, :, None] * B[:, t, :, None, :]
        ys.append(np.einsum("bhpn,bhn->bhp", h, C[:, t]))
    y = np.stack(ys, axis=1)
    if D is not None:
        y = y + x * D[None, None, :, None]
    return y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Bb, L, H, P, N, Q = 1, 256, 4, 64, 64, 64
    x = rng.standard_normal((Bb, L, H, P))
    dt = softplus(rng.standard_normal((Bb, L, H)))
    A = -np.abs(rng.random(H)) - 0.1
    B = rng.standard_normal((Bb, L, H, N))
    C = rng.standard_normal((Bb, L, H, N))
    D = rng.standard_normal(H)
    yc = ssd_chunked(x, dt, A, B, C, D, chunk=Q)
    yn = ssd_naive(x, dt, A, B, C, D)
    err = np.abs(yc - yn).max()
    denom = np.abs(yn).max()
    print(f"chunked vs naive: max abs err = {err:.2e}, rel = {err/denom:.2e}  "
          f"({'OK' if err/denom < 1e-9 else 'MISMATCH'})")
