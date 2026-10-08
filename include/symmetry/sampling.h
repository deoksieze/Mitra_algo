// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/storage.h"
#include <random>
namespace symmetry {
std::vector<Index> sample_vertices(ArrayView<const Scalar> weights,
                                   double ratio, std::mt19937_64 &random);
}
