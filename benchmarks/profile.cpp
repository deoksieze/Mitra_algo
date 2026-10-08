// SPDX-License-Identifier: MPL-2.0
#include "symmetry/cuda.h"
#include "symmetry/io.h"
#include <iomanip>
#include <iostream>
#include <stdexcept>
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("Usage: symmetry_cuda_profile MODEL.off");
    symmetry::take_cuda_metrics();
    auto gpu = symmetry::make_cuda_backend(true);
    symmetry::Options options;
    options.sample_ratio = 1;
    options.pair_threshold = .08;
    options.verify_threshold = .01;
    auto result = symmetry::detect(symmetry::read_mesh(argv[1]), options, *gpu);
    std::cout << std::setprecision(9)
              << "{\"vertices\":" << result.mesh.vertices.size()
              << ",\"faces\":" << result.mesh.faces.size()
              << ",\"selected\":" << result.pruned.size()
              << ",\"pairs\":" << result.pairs.size()
              << ",\"votes\":" << result.votes.size()
              << ",\"clusters\":" << result.clusters.size()
              << ",\"limited_iterations\":" << result.limited_iterations
              << ",\"stages\":[";
    bool first = true;
    for (const auto &m : symmetry::take_cuda_metrics()) {
      if (!m.all_buffers_released)
        throw std::runtime_error("Profile stage leaked buffers");
      if (!first)
        std::cout << ',';
      first = false;
      std::cout << "{\"stage\":\"" << m.stage
                << "\",\"gpu_interval_ms\":" << m.gpu_interval_ms
                << ",\"custom_kernel_ms\":" << m.custom_kernel_ms
                << ",\"custom_launches\":" << m.custom_launches
                << ",\"h2d_bytes\":" << m.h2d_bytes
                << ",\"d2h_bytes\":" << m.d2h_bytes
                << ",\"h2d_ms\":" << m.h2d_ms << ",\"d2h_ms\":" << m.d2h_ms
                << ",\"peak_buffer_bytes\":" << m.peak_buffer_bytes
                << ",\"all_buffers_released\":true}";
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
