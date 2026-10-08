// SPDX-License-Identifier: MPL-2.0
#include "symmetry/io.h"
#include <fstream>
#include <iomanip>
#include <stdexcept>
namespace symmetry {
namespace {
void vec_json(std::ostream &s, Vec v) {
  s << '[' << v.x << ',' << v.y << ',' << v.z << ']';
}
template <class Range> void ids_json(std::ostream &s, const Range &a) {
  s << '[';
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (i)
      s << ',';
    s << a[i];
  }
  s << ']';
}
void patches_json(std::ostream &s, CsrRowsView a) {
  s << '[';
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (i)
      s << ',';
    ids_json(s, a[i]);
  }
  s << ']';
}
std::ofstream output(const std::filesystem::path &p) {
  std::ofstream out(p);
  if (!out)
    throw std::runtime_error("Cannot write " + p.string());
  out << std::setprecision(17);
  return out;
}
} // namespace
void export_result(const Result &r, const std::filesystem::path &directory) {
  std::filesystem::create_directories(directory);
  auto f = output(directory / "features.csv");
  f << "vertex,x,y,z,kmin,kmax,nx,ny,nz,dminx,dminy,dminz,dmaxx,dmaxy,dmaxz,"
       "valid,sampled,pruned\n";
  std::vector<bool> sampled(r.features.size()), pruned(r.features.size());
  for (int i : r.sample)
    sampled[i] = true;
  for (int i : r.pruned)
    pruned[i] = true;
  for (std::size_t i = 0; i < r.features.size(); ++i) {
    auto a = r.features[i];
    Vec p = r.mesh.vertices[i];
    f << i << ',' << p.x << ',' << p.y << ',' << p.z << ',' << a.kmin << ','
      << a.kmax;
    for (Vec v : {a.normal, a.dmin, a.dmax})
      f << ',' << v.x << ',' << v.y << ',' << v.z;
    f << ',' << a.valid << ',' << sampled[i] << ',' << pruned[i] << '\n';
  }
  auto p = output(directory / "pairs.csv");
  p << "pair,a,b\n";
  for (std::size_t i = 0; i < r.pairs.size(); ++i)
    p << i << ',' << r.pairs[i].a << ',' << r.pairs[i].b << '\n';
  auto v = output(directory / "votes.csv");
  v << "vote,pair,type,r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz,nx,ny,nz,d,"
       "cluster\n";
  std::vector<int> assignment(r.votes.size(), -1);
  for (std::size_t c = 0; c < r.clusters.size(); ++c)
    for (int id : r.clusters[c].votes)
      assignment[id] = c;
  for (std::size_t i = 0; i < r.votes.size(); ++i) {
    const auto &a = r.votes[i];
    v << i << ',' << a.pair << ',' << (a.reflection ? "reflection" : "rigid");
    for (double x : a.rotation)
      v << ',' << x;
    v << ',' << a.translation.x << ',' << a.translation.y << ','
      << a.translation.z << ',' << a.plane_normal.x << ',' << a.plane_normal.y
      << ',' << a.plane_normal.z << ',' << a.plane_offset << ','
      << assignment[i] << '\n';
  }
  auto descriptors = output(directory / "descriptors.csv");
  descriptors << "vote,dimension";
  for (int k = 0; k < 12; ++k)
    descriptors << ",component" << k;
  descriptors << '\n';
  for (std::size_t i = 0; i < r.votes.size(); ++i) {
    const auto &d = r.votes[i].descriptor;
    descriptors << i << ',' << d.size();
    for (int k = 0; k < 12; ++k) {
      descriptors << ',';
      if (k < static_cast<int>(d.size()))
        descriptors << d[k];
    }
    descriptors << '\n';
  }
  auto off = output(directory / "mesh.off");
  off << "OFF\n"
      << r.mesh.vertices.size() << ' ' << r.mesh.faces.size() << " 0\n";
  for (Vec x : r.mesh.vertices)
    off << x.x << ' ' << x.y << ' ' << x.z << '\n';
  for (auto face : r.mesh.faces)
    off << "3 " << face[0] << ' ' << face[1] << ' ' << face[2] << '\n';
  auto s = output(directory / "result.json");
  s << "{\"schema_version\":1,\"mode\":\"rigid_and_plane_reflection\",\"box_"
       "size\":"
    << r.box_size << ",\"parameters\":{\"seed\":" << r.options.seed
    << ",\"sample_ratio\":" << r.options.sample_ratio
    << ",\"prune\":" << r.options.prune
    << ",\"pair_threshold\":" << r.options.pair_threshold
    << ",\"bandwidth\":" << r.options.bandwidth
    << ",\"verify_threshold\":" << r.options.verify_threshold
    << ",\"smoothing\":" << r.options.smoothing
    << ",\"subdivide\":" << r.options.subdivide
    << ",\"max_iterations\":" << r.options.max_iterations
    << ",\"max_clusters\":" << r.options.max_clusters
    << ",\"max_pairs\":" << r.options.max_pairs
    << ",\"max_votes\":" << r.options.max_votes
    << ",\"patching\":" << (r.options.patching ? "true" : "false")
    << "},\"counts\":{\"vertices\":" << r.mesh.vertices.size()
    << ",\"faces\":" << r.mesh.faces.size() << ",\"pairs\":" << r.pairs.size()
    << ",\"votes\":" << r.votes.size() << ",\"clusters\":" << r.clusters.size()
    << "},\"diagnostics\":{\"rejected_faces\":" << r.rejected_faces
    << ",\"nonmanifold_edges\":" << r.nonmanifold_edges
    << ",\"orientation_conflicts\":" << r.orientation_conflicts
    << ",\"limited_iterations\":" << r.limited_iterations
    << "},\"timings_ms\":{";
  for (std::size_t i = 0; i < r.timings.size(); ++i) {
    if (i)
      s << ',';
    s << '"' << r.timings[i].first << "\":" << r.timings[i].second;
  }
  s << "},\"vertices\":[";
  for (std::size_t i = 0; i < r.mesh.vertices.size(); ++i) {
    if (i)
      s << ',';
    vec_json(s, r.mesh.vertices[i]);
  }
  s << "],\"faces\":[";
  for (std::size_t i = 0; i < r.mesh.faces.size(); ++i) {
    if (i)
      s << ',';
    auto a = r.mesh.faces[i];
    s << '[' << a[0] << ',' << a[1] << ',' << a[2] << ']';
  }
  s << "],\"sample\":";
  ids_json(s, r.sample);
  s << ",\"pruned\":";
  ids_json(s, r.pruned);
  s << ",\"pairs\":[";
  for (std::size_t i = 0; i < r.pairs.size(); ++i) {
    if (i)
      s << ',';
    s << '[' << r.pairs[i].a << ',' << r.pairs[i].b << ']';
  }
  s << "],\"clusters\":[";
  for (std::size_t i = 0; i < r.clusters.size(); ++i) {
    if (i)
      s << ',';
    const auto &c = r.clusters[i];
    s << "{\"type\":\"" << (c.reflection ? "reflection" : "rigid")
      << "\",\"rotation\":[";
    for (int k = 0; k < 9; ++k) {
      if (k)
        s << ',';
      s << c.rotation[k];
    }
    s << "],\"translation\":";
    vec_json(s, c.translation);
    s << ",\"plane_normal\":";
    vec_json(s, c.plane_normal);
    s << ",\"plane_offset\":" << c.plane_offset << ",\"votes\":";
    ids_json(s, c.votes);
    s << ",\"pair_ids\":[";
    for (std::size_t j = 0; j < c.votes.size(); ++j) {
      if (j)
        s << ',';
      s << r.votes[c.votes[j]].pair;
    }
    s << "],\"coverage\":" << c.coverage << ",\"rms\":" << c.rms
      << ",\"side_a\":";
    ids_json(s, c.side_a);
    s << ",\"side_b\":";
    ids_json(s, c.side_b);
    s << ",\"patches_a\":";
    patches_json(s, c.patches_a);
    s << ",\"patches_b\":";
    patches_json(s, c.patches_b);
    s << '}';
  }
  s << "]}\n";
  if (!f || !p || !v || !descriptors || !off || !s)
    throw std::runtime_error("Output write failed");
}
} // namespace symmetry
