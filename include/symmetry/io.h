// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/pipeline.h"
#include <filesystem>
namespace symmetry {
Mesh read_mesh(const std::filesystem::path &path);
void export_result(const Result &, const std::filesystem::path &directory);
} // namespace symmetry
