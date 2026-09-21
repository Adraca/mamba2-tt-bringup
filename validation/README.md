# Validation harness

Torch-free ttnn PCC harness run on the ttsim functional simulator (release Docker + libttsim_bh.so).
Validates each SSD sub-block (segsum/mask, diagonal block, chunk-state, inter-chunk scan, off-diagonal,
full block, full Mamba2 layer) against `../ssd_ref/ssd_minimal.py` at PCC >= 0.99. Scripts land here as the
ttnn_impl fills in. Every result labelled SIM-CONFIRMED / SIM-ONLY / REQUIRES-SILICON.
