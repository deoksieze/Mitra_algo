// SPDX-License-Identifier: MPL-2.0
#include "symmetry/cuda.h"
#include <iostream>
#include <stdexcept>
using namespace symmetry;
void need(bool c, const char *m) {
  if (!c)
    throw std::runtime_error(m);
}
int main() {
  try {
    take_cuda_metrics();
    Mesh mesh;
    Features f;
    f.resize(8);
    for (int i = 0; i < 8; ++i) {
      mesh.vertices.push_back({double(i), 0, 0});
      f.kmin[i] = 0;
      f.kmax[i] = 1;
      f.valid[i] = 1;
    }
    std::vector<double> weights(8, 1);
    Options o;
    o.sample_ratio = 1;
    auto profiled = make_cuda_backend(true), regular = make_cuda_backend();
    std::mt19937_64 a(42), b(42);
    auto x = profiled->select(mesh.as_view(), f.as_view(), view(weights), o, a);
    auto metrics = take_cuda_metrics();
    auto y = regular->select(mesh.as_view(), f.as_view(), view(weights), o, b);
    need(x.sample == y.sample && x.pruned == y.pruned && a == b,
         "profiling changes results");
    need(take_cuda_metrics().empty(), "disabled profiling records metrics");
    need(metrics.size() == 1 && metrics[0].stage == "sample_prune",
         "stage metric missing");
    auto m = metrics[0];
    need(m.h2d_bytes == 21 * 8 && m.d2h_bytes == 8 + 4 * 8,
         "transfer byte count wrong");
    need(m.peak_buffer_bytes >= 26 * 8 + 8 && m.all_buffers_released,
         "allocation accounting wrong");
    need(m.custom_launches == 1 && m.custom_kernel_ms >= 0 &&
             m.gpu_interval_ms > 0,
         "CUDA events unavailable");
    std::cout << "CUDA profiling passed: exact transfer bytes, peak "
                 "allocations, events, no result changes\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
