// SPDX-License-Identifier: MPL-2.0
#include "symmetry/io.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
namespace symmetry {
Mesh read_mesh(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot open mesh: " + path.string());
  Mesh m;
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  auto triangle = [&](const std::vector<int> &polygon) {
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
      m.faces.push_back({polygon[0], polygon[i], polygon[i + 1]});
  };
  if (ext == ".off") {
    // Strip comments so counts can share the OFF line.
    std::string line, contents;
    while (std::getline(in, line)) {
      line = line.substr(0, line.find('#'));
      contents += line + '\n';
    }
    std::istringstream s(contents);
    std::string magic;
    long long n, f, e;
    s >> magic >> n >> f >> e;
    if (!s || magic != "OFF" || n < 3 || f < 1 || n > 10000000 || f > 20000000)
      throw std::runtime_error("Invalid OFF header");
    m.vertices.resize(n);
    for (Offset i = 0; i < m.vertices.size(); ++i) {
      Vec v;
      if (!(s >> v.x >> v.y >> v.z))
        throw std::runtime_error("Truncated OFF vertices");
      m.vertices.set(i, v);
    }
    for (long long i = 0; i < f; ++i) {
      int count;
      if (!(s >> count) || count < 3 || count > 1000000)
        throw std::runtime_error("Invalid OFF polygon");
      std::vector<int> p(count);
      for (auto &v : p)
        if (!(s >> v))
          throw std::runtime_error("Truncated OFF face");
      triangle(p);
    }
  } else if (ext == ".obj") {
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream s(line);
      std::string tag;
      s >> tag;
      if (tag == "v") {
        Vec v;
        if (!(s >> v.x >> v.y >> v.z))
          throw std::runtime_error("Invalid OBJ vertex");
        m.vertices.push_back(v);
      } else if (tag == "f") {
        std::vector<int> p;
        std::string token;
        while (s >> token) {
          if (token[0] == '#')
            break;
          token = token.substr(0, token.find('/'));
          std::size_t used = 0;
          long long i = std::stoll(token, &used);
          if (used != token.size() || i == 0)
            throw std::runtime_error("Invalid OBJ index");
          long long index =
              i > 0 ? i - 1 : static_cast<long long>(m.vertices.size()) + i;
          if (index < 0 || index >= static_cast<long long>(m.vertices.size()))
            throw std::runtime_error("OBJ index out of range");
          p.push_back(static_cast<int>(index));
        }
        if (p.size() < 3)
          throw std::runtime_error("Invalid OBJ face");
        triangle(p);
      }
    }
  } else
    throw std::runtime_error("Supported mesh formats: OFF and OBJ");
  if (m.vertices.empty() || m.faces.empty())
    throw std::runtime_error("Mesh is empty");
  for (Vec v : m.vertices)
    for (int k = 0; k < 3; ++k)
      if (!std::isfinite(v[k]))
        throw std::runtime_error("Non-finite vertex");
  for (auto f : m.faces)
    for (int i : f)
      if (i < 0 || i >= static_cast<int>(m.vertices.size()))
        throw std::runtime_error("Face index out of range");
  return m;
}
} // namespace symmetry
