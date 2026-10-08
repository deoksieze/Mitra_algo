# CUDA backend (stages 3–8)

The optional backend implements feature pruning, indexed pairing, transformation
votes, mean shift, surface verification, and normals/tensors/tangent eigenframes.
Host code prepares topology/BVH, subdivision, sampling/RNG, seed/merge order,
and completed results. MATLAB and libigl are not required.

Pairing sorts kmin keys, searches conservative ranges, checks candidates in
bounded packets, and uses count → scan → fill. Segmented sorting restores CPU
pair order. Storage is O(S+P) plus CUB scratch; dense ranges can still require
quadratic computation. Output budgets are checked before allocation.

Mean shift evaluates a trajectory in batches of up to eight iterations using one
cooperative block. Component sums retain CPU vote order; support is compacted
once per trajectory. This preserves deterministic seed/merge semantics but
limits parallelism. A CUDA implementation does not imply a speedup.

Verification uses device BVH queries and iterative CSR component labels.
Curvature uses ordered CSR gathers and a tangent-plane 2×2 eigenproblem.
All inputs are borrowed host views; method-owned device buffers return completed
host-owned outputs in CPU order. Precision is double with FMA disabled.

Enable `SYMMETRY_ENABLE_CUDA=ON`, choose `CMAKE_CUDA_ARCHITECTURES` for your GPU,
and run `symmetry_cpu_cli --backend cuda`. The executable name is historical;
CPU-only builds remain supported, and CPU is the default backend.

`make_cuda_backend(true)` enables optional per-stage CUDA event, explicit copy,
custom launch and buffer accounting. Profiling introduces synchronization;
ordinary benchmarks disable it. See [CUDA_STAGE8_RU.md](../../CUDA_STAGE8_RU.md)
for measurement definitions, results and environment limitations, and
[RUNNING_RU.md](../../RUNNING_RU.md) for build/run commands.
