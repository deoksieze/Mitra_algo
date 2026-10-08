// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/math.h"
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace symmetry {
using Scalar = double;
using Index = std::int32_t; // nonnegative IDs; -1 reserved for absence
using Offset = std::uint64_t;
static_assert(sizeof(Index) == 4 && sizeof(Offset) == 8 && sizeof(Scalar) == 8);

// Borrowed HOST memory. No ownership, implicit copying or allocation.
// Views expire when their owner is destroyed, resized or moved.
template <class T> struct ArrayView {
  T *data = nullptr;
  Offset count = 0;
  Offset size() const { return count; }
  bool empty() const { return count == 0; }
  T &operator[](Offset i) const { return data[i]; }
  T *begin() const { return data; }
  T *end() const { return count ? data + count : data; }
};
template <class T> ArrayView<const T> view(const std::vector<T> &v) {
  return {v.data(), static_cast<Offset>(v.size())};
}
inline void require_size(Offset actual, Offset expected) {
  if (actual != expected)
    throw std::invalid_argument("SoA component lengths differ");
}

template <class Owner> struct ValueIterator {
  const Owner *owner;
  Offset index;
  auto operator*() const { return (*owner)[index]; }
  ValueIterator &operator++() {
    ++index;
    return *this;
  }
  bool operator!=(ValueIterator other) const { return index != other.index; }
};
struct PointsView {
  ArrayView<const Scalar> x, y, z;
  Offset size() const { return x.size(); }
  bool empty() const { return x.empty(); }
  Vec operator[](Offset i) const { return {x[i], y[i], z[i]}; }
  Vec front() const { return (*this)[0]; }
  auto begin() const { return ValueIterator<PointsView>{this, 0}; }
  auto end() const { return ValueIterator<PointsView>{this, size()}; }
};
struct Points {
  std::vector<Scalar> x, y, z;
  Offset size() const { return x.size(); }
  bool empty() const { return x.empty(); }
  void resize(Offset n) {
    x.resize(n);
    y.resize(n);
    z.resize(n);
  }
  void set(Offset i, Vec v) {
    x[i] = v.x;
    y[i] = v.y;
    z[i] = v.z;
  }
  void push_back(Vec v) {
    x.push_back(v.x);
    y.push_back(v.y);
    z.push_back(v.z);
  }
  Vec operator[](Offset i) const { return {x[i], y[i], z[i]}; }
  Vec front() const { return (*this)[0]; }
  PointsView as_view() const {
    require_size(y.size(), size());
    require_size(z.size(), size());
    return {view(x), view(y), view(z)};
  }
  auto begin() const { return ValueIterator<Points>{this, 0}; }
  auto end() const { return ValueIterator<Points>{this, size()}; }
};
template <class T, int N> struct ColumnsView {
  std::array<ArrayView<const T>, N> component;
  Offset size() const { return component[0].size(); }
  bool empty() const { return size() == 0; }
  std::array<T, N> operator[](Offset i) const {
    std::array<T, N> r{};
    for (int k = 0; k < N; ++k)
      r[k] = component[k][i];
    return r;
  }
  auto begin() const { return ValueIterator<ColumnsView>{this, 0}; }
  auto end() const { return ValueIterator<ColumnsView>{this, size()}; }
};
template <class T, int N> struct Columns {
  std::array<std::vector<T>, N> component;
  Offset size() const { return component[0].size(); }
  bool empty() const { return size() == 0; }
  void resize(Offset n) {
    for (auto &c : component)
      c.resize(n);
  }
  void push_back(std::array<T, N> row) {
    for (int k = 0; k < N; ++k)
      component[k].push_back(row[k]);
  }
  void set(Offset i, std::array<T, N> row) {
    for (int k = 0; k < N; ++k)
      component[k][i] = row[k];
  }
  std::array<T, N> operator[](Offset i) const {
    std::array<T, N> r{};
    for (int k = 0; k < N; ++k)
      r[k] = component[k][i];
    return r;
  }
  ColumnsView<T, N> as_view() const {
    ColumnsView<T, N> r{};
    for (int k = 0; k < N; ++k) {
      require_size(component[k].size(), size());
      r.component[k] = view(component[k]);
    }
    return r;
  }
  auto begin() const { return ValueIterator<Columns>{this, 0}; }
  auto end() const { return ValueIterator<Columns>{this, size()}; }
};
using Faces = Columns<Index, 3>;
using FacesView = ColumnsView<Index, 3>;
struct MeshView {
  PointsView vertices;
  FacesView faces;
};
struct Mesh {
  Points vertices;
  Faces faces;
  MeshView as_view() const { return {vertices.as_view(), faces.as_view()}; }
};

struct CsrView {
  ArrayView<const Offset> offsets;
  ArrayView<const Index> indices;
  Offset size() const { return offsets.empty() ? 0 : offsets.size() - 1; }
  ArrayView<const Index> operator[](Offset row) const {
    Offset start = offsets[row], length = offsets[row + 1] - start;
    return {length ? indices.data + start : nullptr, length};
  }
};
struct Csr {
  std::vector<Offset> offsets{0};
  std::vector<Index> indices;
  Offset size() const { return offsets.size() - 1; }
  void append(ArrayView<const Index> row) {
    if (row.size() > std::numeric_limits<Offset>::max() - indices.size())
      throw std::overflow_error("CSR size overflow");
    for (Index i : row)
      indices.push_back(i);
    offsets.push_back(indices.size());
  }
  void append(const std::vector<Index> &row) { append(view(row)); }
  static Csr from_rows(const std::vector<std::vector<Index>> &rows) {
    Csr r;
    for (const auto &row : rows)
      r.append(row);
    return r;
  }
  CsrView as_view() const {
    if (offsets.empty() || offsets[0] != 0 ||
        offsets.back() != indices.size() ||
        !std::is_sorted(offsets.begin(), offsets.end()))
      throw std::invalid_argument("Invalid CSR offsets");
    return {view(offsets), view(indices)};
  }
  auto operator[](Offset i) const { return as_view()[i]; }
};
struct CsrRowsView {
  CsrView rows;
  Offset first = 0, count = 0;
  Offset size() const { return count; }
  auto operator[](Offset i) const { return rows[first + i]; }
};
struct GroupedCsr {
  std::vector<Offset> groups{0};
  Csr rows;
  void append(const std::vector<std::vector<Index>> &patches) {
    for (const auto &p : patches)
      rows.append(p);
    groups.push_back(rows.size());
  }
  CsrRowsView operator[](Offset group) const {
    return {rows.as_view(), groups[group], groups[group + 1] - groups[group]};
  }
};

struct Feature {
  Vec normal, dmin, dmax;
  Scalar kmin = 0, kmax = 0;
  bool valid = false;
};
struct FeaturesView {
  PointsView normal, dmin, dmax;
  ArrayView<const Scalar> kmin, kmax;
  ArrayView<const std::uint8_t> valid;
  Offset size() const { return kmin.size(); }
  Feature operator[](Offset i) const {
    return {normal[i], dmin[i], dmax[i], kmin[i], kmax[i], valid[i] != 0};
  }
};
struct Features {
  Points normal, dmin, dmax;
  std::vector<Scalar> kmin, kmax;
  std::vector<std::uint8_t> valid;
  Offset size() const { return kmin.size(); }
  void resize(Offset n) {
    normal.resize(n);
    dmin.resize(n);
    dmax.resize(n);
    kmin.resize(n);
    kmax.resize(n);
    valid.resize(n);
  }
  void set(Offset i, Feature f) {
    normal.set(i, f.normal);
    dmin.set(i, f.dmin);
    dmax.set(i, f.dmax);
    kmin[i] = f.kmin;
    kmax[i] = f.kmax;
    valid[i] = f.valid;
  }
  FeaturesView as_view() const {
    for (Offset n :
         {normal.size(), dmin.size(), dmax.size(),
          static_cast<Offset>(kmax.size()), static_cast<Offset>(valid.size())})
      require_size(n, size());
    return {normal.as_view(), dmin.as_view(), dmax.as_view(),
            view(kmin),       view(kmax),     view(valid)};
  }
  Feature operator[](Offset i) const { return as_view()[i]; }
};
struct Pair {
  Index a, b;
};
struct PairsView {
  ArrayView<const Index> a, b;
  Offset size() const { return a.size(); }
  Pair operator[](Offset i) const { return {a[i], b[i]}; }
};
struct Pairs {
  std::vector<Index> a, b;
  Offset size() const { return a.size(); }
  void push_back(Pair p) {
    a.push_back(p.a);
    b.push_back(p.b);
  }
  PairsView as_view() const {
    require_size(b.size(), size());
    return {view(a), view(b)};
  }
  Pair operator[](Offset i) const { return {a[i], b[i]}; }
};
// Fixed-capacity descriptor: valid components are [0,dimension), tails are
// zero.
struct Descriptor {
  std::array<Scalar, 12> values{};
  std::uint8_t dimension = 0;
  Descriptor() = default;
  Descriptor(std::initializer_list<Scalar> a) {
    if (a.size() > 12)
      throw std::invalid_argument("Descriptor too large");
    dimension = a.size();
    std::copy(a.begin(), a.end(), values.begin());
  }
  Descriptor(const std::vector<Scalar> &a) {
    if (a.size() > 12)
      throw std::invalid_argument("Descriptor too large");
    dimension = a.size();
    std::copy(a.begin(), a.end(), values.begin());
  }
  static Descriptor zero(Offset n) {
    if (n > 12)
      throw std::invalid_argument("Descriptor too large");
    Descriptor r;
    r.dimension = n;
    return r;
  }
  Offset size() const { return dimension; }
  Scalar &operator[](Offset i) { return values[i]; }
  Scalar operator[](Offset i) const { return values[i]; }
  auto begin() { return values.begin(); }
  auto end() { return values.begin() + dimension; }
  auto begin() const { return values.begin(); }
  auto end() const { return values.begin() + dimension; }
};
enum class TransformKind : std::uint8_t { Rigid = 0, Reflection = 1 };
struct Vote {
  Index pair;
  bool reflection;
  Mat rotation;
  Vec translation, plane_normal;
  Scalar plane_offset = 0;
  Descriptor descriptor;
};
struct VotesView {
  ArrayView<const Index> pairs;
  ArrayView<const TransformKind> kind;
  ColumnsView<Scalar, 9> rotation;
  PointsView translation, plane_normal;
  ArrayView<const Scalar> plane_offset;
  ColumnsView<Scalar, 12> descriptor;
  Offset size() const { return pairs.size(); }
  bool is_reflection(Offset i) const {
    return kind[i] == TransformKind::Reflection;
  }
  Descriptor descriptor_at(Offset i) const {
    Descriptor r;
    r.values = descriptor[i];
    r.dimension = is_reflection(i) ? 4 : 12;
    return r;
  }
  Vote operator[](Offset i) const {
    return {pairs[i],        is_reflection(i), rotation[i],     translation[i],
            plane_normal[i], plane_offset[i],  descriptor_at(i)};
  }
};
struct Votes {
  std::vector<Index> pairs;
  std::vector<TransformKind> kind;
  Columns<Scalar, 9> rotation;
  Points translation, plane_normal;
  std::vector<Scalar> plane_offset;
  Columns<Scalar, 12> descriptor;
  Offset size() const { return pairs.size(); }
  void push_back(const Vote &v) {
    if (v.descriptor.size() != (v.reflection ? 4 : 12))
      throw std::invalid_argument("Vote descriptor dimension/type mismatch");
    pairs.push_back(v.pair);
    kind.push_back(v.reflection ? TransformKind::Reflection
                                : TransformKind::Rigid);
    rotation.push_back(v.rotation);
    translation.push_back(v.translation);
    plane_normal.push_back(v.plane_normal);
    plane_offset.push_back(v.plane_offset);
    descriptor.push_back(v.descriptor.values);
  }
  VotesView as_view() const {
    for (Offset n :
         {static_cast<Offset>(kind.size()), rotation.size(), translation.size(),
          plane_normal.size(), static_cast<Offset>(plane_offset.size()),
          descriptor.size()})
      require_size(n, size());
    return {view(pairs),
            view(kind),
            rotation.as_view(),
            translation.as_view(),
            plane_normal.as_view(),
            view(plane_offset),
            descriptor.as_view()};
  }
  Vote operator[](Offset i) const { return as_view()[i]; }
};
struct ClusterRecord {
  bool reflection = false;
  Mat rotation{};
  Vec translation, plane_normal;
  Scalar plane_offset = 0;
};
struct Verification {
  Csr side_a, side_b;
  GroupedCsr patches_a, patches_b;
  std::vector<Scalar> coverage, rms;
  void validate(Offset count) const {
    require_size(side_a.as_view().size(), count);
    require_size(side_b.as_view().size(), count);
    require_size(coverage.size(), count);
    require_size(rms.size(), count);
    for (const GroupedCsr *p : {&patches_a, &patches_b}) {
      require_size(p->groups.size(), count + 1);
      if (p->groups.front() != 0 ||
          p->groups.back() != p->rows.as_view().size() ||
          !std::is_sorted(p->groups.begin(), p->groups.end()))
        throw std::invalid_argument("Invalid patch group offsets");
    }
  }
  static Verification empty(Offset count) {
    Verification r;
    r.coverage.resize(count);
    r.rms.resize(count);
    for (Offset i = 0; i < count; ++i) {
      r.side_a.append(std::vector<Index>{});
      r.side_b.append(std::vector<Index>{});
      r.patches_a.append({});
      r.patches_b.append({});
    }
    return r;
  }
};
struct ClusterView : ClusterRecord {
  ArrayView<const Index> votes, side_a, side_b;
  CsrRowsView patches_a, patches_b;
  Scalar coverage = 0, rms = 0;
};
struct ClustersView {
  ArrayView<const ClusterRecord> records;
  CsrView votes;
  Offset size() const { return records.size(); }
};
struct Clusters {
  std::vector<ClusterRecord> records;
  Csr votes;
  Verification verification;
  Offset size() const { return records.size(); }
  bool empty() const { return records.empty(); }
  void append(ClusterRecord r, const std::vector<Index> &members) {
    records.push_back(r);
    votes.append(members);
  }
  ClustersView as_view() const {
    require_size(votes.size(), size());
    return {view(records), votes.as_view()};
  }
  ClusterView operator[](Offset i) const {
    ClusterView r;
    static_cast<ClusterRecord &>(r) = records[i];
    r.votes = votes[i];
    r.side_a = verification.side_a[i];
    r.side_b = verification.side_b[i];
    r.patches_a = verification.patches_a[i];
    r.patches_b = verification.patches_b[i];
    r.coverage = verification.coverage[i];
    r.rms = verification.rms[i];
    return r;
  }
};
static_assert(std::is_trivially_copyable<PointsView>::value);
static_assert(std::is_trivially_copyable<FeaturesView>::value);
static_assert(std::is_trivially_copyable<VotesView>::value);
static_assert(std::is_trivially_copyable<CsrView>::value);
} // namespace symmetry
