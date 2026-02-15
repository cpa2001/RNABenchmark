# EcoRNA Model Adapter

This directory contains the RNABenchmark adapter for EcoRNA (`modeling_ecorna.py`).
It wraps the EcoRNA pretrained checkpoint into the RNABenchmark model interface used by downstream BEACON tasks.

## Requirements

Use a modern PyTorch/Transformers stack compatible with the EcoRNA training/inference codebase:

- `torch>=2.2`
- `transformers>=4.40`
- `accelerate>=0.27`
- `tokenizers>=0.15`
- `safetensors>=0.4`

For EcoRNA checkpoints trained with fused kernels:

- `liger-kernel` (required to fully match the original model path)
- `flash-attn` (strongly recommended for fast attention execution)

If optional fused kernels are unavailable, execution may still run via fallback implementations, but exact behavior/speed can differ from the original setup.

## Inference Controls

EcoRNA supports test-time looped transformer execution and multiple sequence pooling strategies through environment variables consumed by the RNABenchmark scripts:

- `ECORNA_NUM_LOOPS`: loop count at inference (for test-time scaling studies)
- `ECORNA_POOLING_STRATEGY`: sequence embedding strategy (for example `cls`, `cls_tanh`, `mean`, `mix`)
- `ECORNA_CHECKPOINT`: path to the pretrained checkpoint

Typical entrypoint for NoncodingRNAFamily:

```bash
bash scripts/opensource/run_ncrna.sh ecorna
```

See the repository root `README.md` for end-to-end command examples.
