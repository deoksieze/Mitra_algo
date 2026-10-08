# Hybrid CUDA backend (stage 3)

`backend.cu` implements GPU feature pruning with stable CUB compaction and GPU
transformation votes with count → exclusive scan → fill. Curvature, sampling,
pairing, clustering and verification currently run on CPU.

All Backend inputs are borrowed host views. Device buffers belong to each method;
completed host-owned outputs preserve CPU ordering and the export format.

Enable with `-DSYMMETRY_ENABLE_CUDA=ON`, set `CMAKE_CUDA_ARCHITECTURES` for your GPU,
and run the CLI with `--backend cuda`. CPU-only builds keep CUDA optional.

See [CUDA_STAGE3_RU.md](../../CUDA_STAGE3_RU.md) for tested build commands,
CPU/GPU comparisons, timings and the Compute Sanitizer environment limitation.
