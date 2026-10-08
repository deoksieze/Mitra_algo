# Hybrid CUDA backend (stages 3–4)

`backend.cu` implements GPU feature pruning, indexed GPU pairing and GPU
transformation votes. Curvature, sampling, clustering and verification run on CPU.

Pairing sorts kmin keys, searches conservative candidate ranges, checks the CPU
predicate in bounded packets, and uses count → exclusive scan → fill. Segmented
sorting of accepted target ordinals restores CPU pair order. Storage is O(S+P)
plus CUB scratch, without an S² candidate matrix. Dense ranges can still require
quadratic computation; the output budget is checked before pair allocation.

All Backend inputs are borrowed host views. Device buffers belong to each method;
completed host-owned outputs preserve CPU ordering and the export format.

Enable with `-DSYMMETRY_ENABLE_CUDA=ON`, set `CMAKE_CUDA_ARCHITECTURES` for your GPU,
and run the CLI with `--backend cuda`. CPU-only builds keep CUDA optional.

See [CUDA_STAGE3_RU.md](../../CUDA_STAGE3_RU.md) and
[CUDA_STAGE4_RU.md](../../CUDA_STAGE4_RU.md) for build/run commands, tests,
measurements and the Compute Sanitizer environment limitation.
