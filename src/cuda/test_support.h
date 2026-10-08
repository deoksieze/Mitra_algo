// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/math.h"
#include <vector>
namespace symmetry::cuda_detail {
struct TriangleQuery { Vec p,a,b,c; };
std::vector<double> triangle_distances(const std::vector<TriangleQuery> &);
}
