# Mamba-2 SSD — math, chunked algorithm, and the Tensix mapping

## 1. The SSD recurrence
Per (batch, head), the Mamba-2 State-Space-Duality layer is a linear time-varying SSM with a **scalar**
state-transition (the key SSD simplification vs. Mamba-1's diagonal `A`):

    h_t = a_t · h_{t-1} + B_t xᵀ_t          h_t ∈ ℝ^{N×P}   (N = state dim, P = head dim)
    y_t = C_tᵀ h_t + D · x_t                y_t ∈ ℝ^{P}

with the data-dependent decay `a_t = exp(Δ_t · A)`, `A` a scalar (per head) < 0 ⇒ `a_t ∈ (0,1)`, and
`B_t, C_t ∈ ℝ^{N}`, `x_t ∈ ℝ^{P}`. `Δ_t` is the input-dependent discretization step (softplus of a projection).

## 2. Duality: the recurrence IS an attention matrix
Unrolling gives `y = M x` with the **1-semiseparable** matrix

    M_{ij} = C_iᵀ ( a_i a_{i-1} … a_{j+1} ) B_j   for i ≥ j,   0 otherwise
           = L_{ij} · (C B^T)_{ij},     L_{ij} = exp( Σ_{k=j+1}^{i} log a_k )  = exp(segsum(log a))

So SSD = **structured masked attention**: an element-wise product of a causal decay mask `L` (a "segment-sum"
of `log a`) with the score matrix `C Bᵀ`, applied to `x`. This is the matmul form — the reason Mamba-2 maps
to the Tensix matrix engine, unlike Mamba-1's scalar sequential scan.

## 3. The chunked algorithm (O(T·N·P), the one we implement)
Naïve `M x` is O(T²). Split the length-T sequence into C chunks of length Q (T = C·Q) and compute four terms:

1. **Diagonal (intra-chunk) blocks** — for each chunk c, the Q×Q masked attention:
   `Y_diag[c] = ( (C[c] B[c]ᵀ) ∘ L[c] ) · X[c]`  where `L[c] = exp(segsum(log a[c]))` is Q×Q, lower-triangular.
   Pure matmul + a triangular decay mask. This is the dominant compute.

2. **Chunk final states** — each chunk collapses to an N×P state carrying its contribution forward:
   `S[c] = Σ_{t∈c} (a_{end} / a_{≤t}) · B_t x_tᵀ`  = `(decay_weights ⊙ B[c])ᵀ · X[c]`.

3. **Inter-chunk recurrence** — run the *scalar* recurrence over just the C chunk-states (C ≪ T ⇒ cheap):
   `Ŝ[c] = α_c · Ŝ[c-1] + S[c-1]`, `α_c` = total decay of chunk c-1.

4. **Off-diagonal (state→output) blocks** — each chunk reads the carried-in state:
   `Y_off[c] = ( C[c] ∘ decay_from_chunk_start ) · Ŝ[c]`.

   `Y = Y_diag + Y_off + D ⊙ X`.

### Op inventory (all present in ttnn)
- `log a`, `exp` — SFPU elementwise; `Δ = softplus(...)`.
- **segsum** — cumulative sum along the chunk axis (`ttnn` cumsum) → triangular decay `L`.
- matmuls: `C Bᵀ` (Q×N·N×Q), masked `·X` (Q×Q·Q×P), state builds and reads (N×P). **Matmul-dominant.**
- Hadamard masks (`L`, decay weights), triangular masking, residual adds.

## 4. Tensix / tt-metal mapping notes
- Chunk length **Q = 64 or 128** aligns to tile (32×32) boundaries and keeps the Q×Q block in L1.
- The decay mask `L` is precomputed per chunk from `segsum(log a)` — a cumsum + `exp`; guard `exp` overflow by
  the standard `exp(segsum - segsum.max)` shift (a numerics trap to PCC-validate explicitly).
- State tensors are N×P per head (N≈64–128, P≈64) — small, stay resident; the inter-chunk scan is a short
  sequential loop over C chunk-states (not per-token) → not bandwidth-bound.
- DEST/accumulation: run matmul accumulation in fp32 (`fp32_dest_acc_en`) where the decay products lose bits;
  validate bf16-vs-fp32 PCC to size the real requirement.

## 5. Validation plan (ttsim, $0)
Golden = torch `ssd_minimal_discrete` (from the official Mamba-2 reference), plus the full `Mamba2` mixer.
Validate bottom-up, each **SIM-CONFIRMED** at PCC ≥ 0.99:
1. segsum + `L` mask alone; 2. single-chunk diagonal block; 3. chunk-state build; 4. inter-chunk scan;
5. off-diagonal read; 6. full multi-chunk SSD block; 7. full `Mamba2` layer (in/out projections + conv + norm).
Config: iterate at a small real config (mamba2-130m) then confirm at a serving config (mamba2-2.7b).
Perf/throughput and full autoregressive serve are **REQUIRES-SILICON** (ttsim is not a timing oracle).

## 6. References (to pin exactly in `ssd_ref/`)
- Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space
  Duality* (Mamba-2). The `ssd_minimal_discrete` chunked reference is the golden.
- Existing TT Mamba-1 demo (`models/demos/wormhole/mamba`) for tokenizer/weight-load/serve plumbing to reuse.
