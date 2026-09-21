# Sim-validation results (ttsim functional simulator, Blackhole)

All results **SIM-CONFIRMED** = real ttnn op-graph on `libttsim` (bit-exact arithmetic), PCC vs the fp64
numpy golden (`../ssd_ref/ssd_minimal.py`). Run in the tt-metal release Docker image, $0, no hardware.
Gate = TT's standard bring-up threshold **PCC ≥ 0.99**.

| Milestone | What | PCC | Status |
|---|---|---|---|
| **M1** | Single-chunk SSD **diagonal block** `Ydiag = (C·Bᵀ ∘ L)·x` — the core SSD-as-structured-masked-attention, as a ttnn matmul→mul→matmul graph | **1.000000** | ✅ SIM-CONFIRMED |
| M1b | Decay-mask identity `L = tril(exp(segsum(a))) == tril(exp(cumsum diff))` (device-cumsum path is valid) | 1.000000 | ✅ verified |
| **M2** | **Full multi-chunk SSD scan** — all four terms (diagonal `Ydiag`, off-diagonal state-read `Yoff`, chunk-state build, inter-chunk recurrence) as a ttnn op-graph, L=128 / 2 chunks | **1.000000** | ✅ SIM-CONFIRMED |
| **M3** | **Full `Mamba2` mixer block** — in_proj + per-head SSD scan + **gated RMSNorm** + out_proj (ttnn); conv1d + softplus + splits on host this milestone | **0.999987** | ✅ SIM-CONFIRMED |
| M3b | Move causal conv1d + softplus(dt) on-device | — | next |
| — | On-silicon forward + throughput | — | REQUIRES-SILICON |

## M1 detail
Config: Q=64 (tile-aligned chunk), N=64 state dim, P=64 head dim, single batch/head/chunk.
The single-chunk case is exactly the diagonal block (carried-in state = 0 for chunk 0), so it isolates the
SSD duality core: scores `C·Bᵀ`, Hadamard with the causal decay mask `L`, then `·(Δ·x)`. All three are
native ttnn tile ops. PCC 1.000000 (bit-exact) on the simulator. Sim wall time ~31 s under x86 emulation.

Reproduce: `validation/` scripts + release Docker + `libttsim`; see repo README.
