# Mamba-2 / SSD on Tenstorrent

Hardware-free bring-up of the **Mamba-2 State-Space-Duality (SSD)** layer on Tenstorrent (Blackhole),
validated on the `ttsim` functional simulator against a PyTorch reference at the **PCC ≥ 0.99** bring-up gate.

## Why Mamba-2 (and why it's a good TT fit)
tt-metal today ships only **Mamba-1** (`models/demos/wormhole/mamba`, the 2.8B selective-scan model), whose
sequential associative scan is memory-bandwidth-bound — reflected in its open perf/CI issues. **Mamba-2 is
different by design:** its SSD formulation re-expresses the same linear state-space recurrence as a sequence of
**matmuls** (structured masked attention over chunks). Matmul is exactly what the Tensix matrix engine is built
for, so Mamba-2 is a *better* architectural fit for TT than the Mamba-1 demo it complements — and the SSD
chunk kernel is a reusable substrate for the whole 2025–26 hybrid-SSM family (Falcon-H1, Granite-4.0-H,
Codestral-Mamba, Zamba), none of which are on TT yet.

## Status
- [ ] SSD reference (torch `ssd_minimal` + full `Mamba2` block) pinned as the golden
- [ ] Chunked-SSD ttnn op graph (segsum via cumsum, masked matmul, chunk-state recurrence)
- [x] M1 diagonal block SIM-CONFIRMED (PCC 1.000000); per-subblock then full block (target ≥ 0.99)
- [ ] Model integration (`models/experimental/mamba2/`) + weight loading from `state-spaces/mamba2-*`
- [ ] Packaging (tt-cli / tt-model-manager bundle)
- [ ] On-silicon forward + perf (REQUIRES-SILICON — pending cloud/hardware access)

Every result is labelled **SIM-CONFIRMED** (PCC on ttsim), **SIM-ONLY** (sim can't be an oracle for it), or
**REQUIRES-SILICON** (perf/throughput/serve). ttsim is bit-exact for arithmetic but not for timing/RNG/perf.

## Layout
- `design/` — the SSD math derivation + the chunked-scan algorithm + the TT mapping (the core doc).
- `ssd_ref/` — pinned torch reference (`ssd_minimal_discrete`) used as the golden.
- `ttnn_impl/` — the chunked-SSD implemented as a ttnn op graph.
- `validation/` — torch-free ttnn PCC harness run on the ttsim functional simulator ($0, no card).

## Reproduce the sim validation ($0, no hardware)
tt-metal release Docker image + `libttsim_bh.so`; `soc_descriptor.yaml` in CWD; then run the
`validation/` scripts under `TT_METAL_SIMULATOR=libttsim_bh.so`.
