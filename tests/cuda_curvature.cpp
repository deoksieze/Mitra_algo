// SPDX-License-Identifier: MPL-2.0
#include "symmetry/cuda.h"
#include "symmetry/io.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace symmetry;
void need(bool b,const char *m) {if(!b) throw std::runtime_error(m);}
void close(double a,double b) {need(std::isfinite(b)&&std::abs(a-b)<1e-10*(1+std::abs(a)),"curvature scalar differs");}
int main(int argc,char **argv) {
  try {
    need(argc==2,"repository root required");auto cpu=make_cpu_backend(),gpu=make_cuda_backend();
    for(const char *name:{"reflection","rigid","partial","single"}) for(int smooth:{0,3,7}) {
      Options o;o.smoothing=smooth;
      auto p=prepare(read_mesh(std::filesystem::path(argv[1])/"examples"/(std::string(name)+".off")),o);
      auto a=cpu->curvature(p.mesh.as_view(),p.topology.as_view(),o);
      auto b=gpu->curvature(p.mesh.as_view(),p.topology.as_view(),o);
      need(a.valid==b.valid,"curvature validity differs");
      for(Offset i=0;i<a.size();++i) {
        auto x=a[i],y=b[i];close(x.kmin,y.kmin);close(x.kmax,y.kmax);
        for(int k=0;k<3;++k) close(x.normal[k],y.normal[k]);
        if(!x.valid) continue;
        need(std::abs(dot(x.dmin,y.dmin))>1-1e-10 && std::abs(dot(x.dmax,y.dmax))>1-1e-10,"principal direction differs beyond sign");
        close(norm(y.dmin),1);close(norm(y.dmax),1);close(dot(y.normal,y.dmin),0);close(dot(y.normal,y.dmax),0);
        need(y.kmin<=y.kmax,"eigenvalues out of order");
      }
      auto again=gpu->curvature(p.mesh.as_view(),p.topology.as_view(),o);
      need(b.kmin==again.kmin && b.dmin.x==again.dmin.x,"curvature repeat differs");
    }
    Mesh plane;plane.vertices.push_back({0,0,0});plane.vertices.push_back({1,0,0});plane.vertices.push_back({1,1,0});plane.vertices.push_back({0,1,0});plane.faces.push_back({0,1,2});plane.faces.push_back({0,2,3});
    Options o;auto p=prepare(std::move(plane),o);auto planar=gpu->curvature(p.mesh.as_view(),p.topology.as_view(),o);
    for(Offset i=0;i<planar.size();++i) {auto f=planar[i];need(!f.valid && f.kmin==0 && f.kmax==0,"planar curvature unstable");}
    std::cout<<"CUDA curvature passed: normals, kmin/kmax, sign-invariant frames, smoothing 0/3/7, boundaries, plane, repeat\n";
    return 0;
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
