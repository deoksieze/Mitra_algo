// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/pipeline.h"
namespace symmetry {
// Requires Symmetry::cuda. Throws if no CUDA device is available.
std::unique_ptr<Backend> make_cuda_backend();
} // namespace symmetry
