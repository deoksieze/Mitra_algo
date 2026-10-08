// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/pipeline.h"
namespace symmetry {
// Requires Symmetry::cuda. Throws if no CUDA device is available.
std::unique_ptr<Backend> make_cuda_backend(bool profile = false);
struct CudaMetric {
  std::string stage;
  std::uint64_t h2d_bytes = 0, d2h_bytes = 0, peak_buffer_bytes = 0,
                custom_launches = 0;
  double gpu_interval_ms = 0, h2d_ms = 0, d2h_ms = 0, custom_kernel_ms = 0;
  bool all_buffers_released = false;
};
// Profiling is opt-in; metrics are collected on the invoking host thread.
std::vector<CudaMetric> take_cuda_metrics();
} // namespace symmetry
