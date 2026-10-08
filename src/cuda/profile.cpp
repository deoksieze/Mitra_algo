// SPDX-License-Identifier: MPL-2.0
#include "profile.h"
#include "symmetry/cuda.h"
#include <algorithm>
#include <stdexcept>
namespace symmetry {
namespace cuda_detail {
namespace {
struct State {
  CudaMetric metric;
  std::size_t live = 0;
  cudaEvent_t start = nullptr, stop = nullptr, copy_start = nullptr,
              copy_stop = nullptr, kernel_start = nullptr,
              kernel_stop = nullptr;
  State *previous = nullptr;
  ~State() {
    for (auto event :
         {start, stop, copy_start, copy_stop, kernel_start, kernel_stop})
      if (event)
        cudaEventDestroy(event);
  }
};
thread_local State *active = nullptr;
thread_local std::vector<CudaMetric> metrics;
void need(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace
ProfileStage::ProfileStage(const char *stage, bool enabled) {
  if (!enabled)
    return;
  auto candidate = std::make_unique<State>();
  candidate->metric.stage = stage;
  candidate->previous = active;
  for (auto event : {&candidate->start, &candidate->stop,
                     &candidate->copy_start, &candidate->copy_stop,
                     &candidate->kernel_start, &candidate->kernel_stop})
    need(cudaEventCreate(event));
  need(cudaEventRecord(candidate->start));
  active = candidate.release();
  state = active;
}
ProfileStage::~ProfileStage() {
  if (!state)
    return;
  std::unique_ptr<State> owned(static_cast<State *>(state));
  active = owned->previous;
  float ms = 0;
  if (cudaEventRecord(owned->stop) == cudaSuccess &&
      cudaEventSynchronize(owned->stop) == cudaSuccess &&
      cudaEventElapsedTime(&ms, owned->start, owned->stop) == cudaSuccess) {
    owned->metric.gpu_interval_ms = ms;
    owned->metric.all_buffers_released = owned->live == 0;
    try {
      metrics.push_back(owned->metric);
    } catch (...) {
    }
  }
}
void allocated(std::size_t bytes) {
  if (active) {
    active->live += bytes;
    active->metric.peak_buffer_bytes =
        std::max<std::uint64_t>(active->metric.peak_buffer_bytes, active->live);
  }
}
void released(std::size_t bytes) {
  if (active)
    active->live -= bytes;
}
cudaError_t profile_copy(void *target, const void *source, std::size_t bytes,
                         cudaMemcpyKind kind) {
  if (!active)
    return cudaMemcpy(target, source, bytes, kind);
  need(cudaEventRecord(active->copy_start));
  auto result = cudaMemcpy(target, source, bytes, kind);
  if (result != cudaSuccess)
    return result;
  need(cudaEventRecord(active->copy_stop));
  need(cudaEventSynchronize(active->copy_stop));
  float ms = 0;
  need(cudaEventElapsedTime(&ms, active->copy_start, active->copy_stop));
  if (kind == cudaMemcpyHostToDevice) {
    active->metric.h2d_bytes += bytes;
    active->metric.h2d_ms += ms;
  }
  if (kind == cudaMemcpyDeviceToHost) {
    active->metric.d2h_bytes += bytes;
    active->metric.d2h_ms += ms;
  }
  return result;
}
void begin_kernel() {
  if (active)
    need(cudaEventRecord(active->kernel_start));
}
void end_kernel() {
  if (!active)
    return;
  need(cudaEventRecord(active->kernel_stop));
  need(cudaEventSynchronize(active->kernel_stop));
  float ms = 0;
  need(cudaEventElapsedTime(&ms, active->kernel_start, active->kernel_stop));
  active->metric.custom_kernel_ms += ms;
  ++active->metric.custom_launches;
}
std::vector<CudaMetric> take_metrics() {
  auto result = std::move(metrics);
  metrics.clear();
  return result;
}
} // namespace cuda_detail
std::vector<CudaMetric> take_cuda_metrics() {
  return cuda_detail::take_metrics();
}
} // namespace symmetry
