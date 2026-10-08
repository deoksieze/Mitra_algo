// SPDX-License-Identifier: MPL-2.0
#pragma once
#include <cstddef>
#include <cuda_runtime.h>
namespace symmetry::cuda_detail {
class ProfileStage {
  void *state = nullptr;

public:
  ProfileStage(const char *, bool);
  ~ProfileStage();
  ProfileStage(const ProfileStage &) = delete;
  ProfileStage &operator=(const ProfileStage &) = delete;
};
void allocated(std::size_t);
void released(std::size_t);
cudaError_t profile_copy(void *, const void *, std::size_t, cudaMemcpyKind);
void begin_kernel();
void end_kernel();
} // namespace symmetry::cuda_detail
