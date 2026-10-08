// SPDX-License-Identifier: MPL-2.0
#include "symmetry/cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace symmetry;
int main() {
  try {
    auto cpu = make_cpu_backend(), gpu = make_cuda_backend();
    Options o;
    o.bandwidth = .03;
    o.max_clusters = 4;
    std::cout << "[";
    bool first = true;
    for (int count : {1024, 4096, 16384}) {
      Votes votes;
      for (int i = 0; i < count; ++i) {
        bool reflection = i % 2;
        double d = 3 + (i % 4 / 2) * 2 + .001 * std::sin(i);
        Vote v{};
        v.pair = i;
        v.reflection = reflection;
        v.rotation = reflection ? reflection_matrix({1, 0, 0}) : identity();
        v.translation = {reflection ? -2 * d : d, 0, 0};
        v.plane_normal = reflection ? Vec{1, 0, 0} : Vec{};
        v.plane_offset = reflection ? d : 0;
        v.descriptor =
            reflection ? Descriptor{1, 0, 0, d / 10} : Descriptor::zero(12);
        if (!reflection) {
          for (int k = 0; k < 9; ++k)
            v.descriptor[k] = v.rotation[k] / 2;
          v.descriptor[9] = d / 10;
        }
        votes.push_back(v);
      }
      auto run = [&](const Backend &backend) {
        std::mt19937_64 rng(42);
        return backend.clustering(votes.as_view(), {10, {}}, o, rng);
      };
      auto reference = run(*cpu);
      run(*gpu);
      double median[2];
      int k = 0;
      for (const Backend *backend : {cpu.get(), gpu.get()}) {
        std::vector<double> times;
        for (int repeat = 0; repeat < 3; ++repeat) {
          auto start = std::chrono::steady_clock::now();
          auto result = run(*backend);
          times.push_back(std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count());
          if (reference.clusters.votes.indices !=
                  result.clusters.votes.indices ||
              reference.clusters.votes.offsets != result.clusters.votes.offsets)
            throw std::runtime_error("Cluster benchmark membership differs");
        }
        std::sort(times.begin(), times.end());
        median[k++] = times[1];
      }
      auto measured = make_cuda_backend(true);
      take_cuda_metrics();
      run(*measured);
      auto metrics = take_cuda_metrics();
      if (metrics.size() != 1 || !metrics[0].all_buffers_released)
        throw std::runtime_error("Cluster metric accounting failed");
      const auto &m = metrics[0];
      if (!first)
        std::cout << ',';
      first = false;
      std::cout << "{\"votes\":" << count
                << ",\"clusters\":" << reference.clusters.size()
                << ",\"cpu_median_ms\":" << median[0]
                << ",\"cuda_median_ms\":" << median[1]
                << ",\"peak_buffer_bytes\":" << m.peak_buffer_bytes
                << ",\"h2d_bytes\":" << m.h2d_bytes
                << ",\"d2h_bytes\":" << m.d2h_bytes
                << ",\"custom_launches\":" << m.custom_launches << "}";
    }
    std::cout << "]\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
