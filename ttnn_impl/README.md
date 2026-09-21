# ttnn implementation

The chunked-SSD op graph in ttnn (segsum via cumsum, masked matmul C·Bᵀ∘L, chunk-state builds/reads,
short inter-chunk scan). Built bottom-up and PCC-validated per sub-block on ttsim before assembly.
See ../design/SSD_MATH_AND_TT_MAPPING.md for the op inventory and Tensix mapping.
