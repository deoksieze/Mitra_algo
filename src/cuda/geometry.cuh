// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "internal.cuh"
namespace symmetry::cuda_detail {
__device__ inline double segment_distance2(V p, V a, V b) {
  V d = b - a;
  double l = dotv(d, d);
  double t = l > 1e-14 ? fmin(1.0, fmax(0.0, dotv(p - a, d) / l)) : 0;
  V e = p - (a + d * t);
  return dotv(e, e);
}
__device__ inline double point_triangle_distance2(V p, V a, V b, V c) {
  V ab = b - a, ac = c - a, ap = p - a;
  if (dotv(crossv(ab, ac), crossv(ab, ac)) < 1e-14 * 1e-14)
    return fmin(segment_distance2(p, a, b),
                fmin(segment_distance2(p, a, c), segment_distance2(p, b, c)));
  double d1 = dotv(ab, ap), d2 = dotv(ac, ap);
  if (d1 <= 0 && d2 <= 0)
    return dotv(ap, ap);
  V bp = p - b;
  double d3 = dotv(ab, bp), d4 = dotv(ac, bp);
  if (d3 >= 0 && d4 <= d3)
    return dotv(bp, bp);
  double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    V e = p - (a + ab * (d1 / (d1 - d3)));
    return dotv(e, e);
  }
  V cp = p - c;
  double d5 = dotv(ab, cp), d6 = dotv(ac, cp);
  if (d6 >= 0 && d5 <= d6)
    return dotv(cp, cp);
  double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    V e = p - (a + ac * (d2 / (d2 - d6)));
    return dotv(e, e);
  }
  double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    V e = p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))));
    return dotv(e, e);
  }
  V e = p - (a + ab * (vb / (va + vb + vc)) + ac * (vc / (va + vb + vc)));
  return fmax(0.0, dotv(e, e));
}
} // namespace symmetry::cuda_detail
