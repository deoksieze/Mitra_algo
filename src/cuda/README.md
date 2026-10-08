# Future CUDA backend

Stage 2 provides `symmetry::Backend` and SoA/CSR host data in `include/symmetry/`.
No CUDA implementation is registered or built yet. Stage 3 will implement
feature/pruning and transformation kernels behind this interface. Curvature
estimation will initially remain on the CPU.

The backend must own its device allocations and explicit host/device transfers.
Host views never own memory and must not be interpreted as device pointers.
Start with double, int32 IDs and uint64 offsets; return host-owned results with
identical stage ordering and indexing. Keep transfers outside per-element loops.
