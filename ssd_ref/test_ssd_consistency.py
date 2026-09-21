"""
test_ssd_consistency.py — adversarial self-consistency sweep for the SSD golden.

Directly answers the audit concern "does chunked==naive only by luck of dimensions / seed?".
Runs ssd_chunked vs ssd_naive over many randomized configs: varied batch/heads/seqlen/state/head-dim,
multiple chunk sizes, and a wide range of decay magnitudes (including large-|A| that would expose any
segsum / state-accumulation / decay-direction error). If every config matches to fp64 machine precision,
the segsum + chunk-state + inter-chunk recurrence are correct (refuting the static "CRITICAL" claims).
"""
import numpy as np
from ssd_minimal import ssd_chunked, ssd_naive, softplus

def one(seed, Bb, H, nchunks, Q, N, P, Ascale):
    rng = np.random.default_rng(seed)
    L = nchunks * Q
    x = rng.standard_normal((Bb, L, H, P))
    dt = softplus(rng.standard_normal((Bb, L, H)))
    A = -(rng.random(H) + 0.05) * Ascale        # negative decays; Ascale stresses magnitude
    B = rng.standard_normal((Bb, L, H, N))
    C = rng.standard_normal((Bb, L, H, N))
    D = rng.standard_normal(H)
    yc = ssd_chunked(x, dt, A, B, C, D, chunk=Q)
    yn = ssd_naive(x, dt, A, B, C, D)
    denom = np.abs(yn).max() + 1e-12
    return np.abs(yc - yn).max() / denom

def main():
    configs = [
        # seed, B, H, nchunks, Q, N, P, Ascale
        (0, 1, 1, 1, 64, 64, 64, 1.0),
        (1, 1, 4, 2, 64, 64, 64, 1.0),
        (2, 2, 8, 3, 32, 128, 64, 1.0),
        (3, 1, 2, 4, 16, 64, 32, 3.0),      # small chunks, larger decay
        (4, 3, 6, 2, 64, 32, 128, 5.0),     # large |A| — stresses exp/segsum
        (5, 1, 1, 5, 8, 16, 16, 10.0),      # very large decay, many tiny chunks
        (6, 2, 4, 1, 128, 64, 64, 0.3),     # single big chunk, gentle decay
        (7, 1, 3, 3, 48, 96, 48, 2.0),      # non-power-of-two dims
    ]
    worst = 0.0; fails = 0
    for cfg in configs:
        r = one(*cfg)
        worst = max(worst, r)
        ok = r < 1e-9
        fails += (not ok)
        print(f"  seed={cfg[0]} B={cfg[1]} H={cfg[2]} chunks={cfg[3]} Q={cfg[4]} N={cfg[5]} P={cfg[6]} "
              f"Ascale={cfg[7]:<4}  rel_err={r:.2e}  {'OK' if ok else 'FAIL'}")
    print(f"\n  worst rel_err over {len(configs)} configs = {worst:.2e}  "
          f"-> {'ALL CONSISTENT (segsum+state+scan correct)' if fails == 0 else f'{fails} FAILED'}")

if __name__ == "__main__":
    main()
