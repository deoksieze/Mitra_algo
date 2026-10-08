// SPDX-License-Identifier: MPL-2.0
#include "symmetry/cuda.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace symmetry;
void need(bool c,const char *m) {if(!c) throw std::runtime_error(m);}
void close(double a,double b) {need(std::isfinite(b)&&std::abs(a-b)<1e-10*(1+std::abs(a)),"cluster center differs");}
int main() {
  try {
    auto cpu=make_cpu_backend(),gpu=make_cuda_backend();
    for(int count:{0,1,64,2048}) {
      Votes votes;
      for(int i=0;i<count;++i) {
        bool reflection=i%2;double d=3+(i%4/2)*2+.001*std::sin(i);
        Vote v{};v.pair=i;v.reflection=reflection;
        v.rotation=reflection ? reflection_matrix({1,0,0}):identity();
        v.translation={reflection ? -2*d : d,0,0};
        v.plane_normal=reflection ? Vec{1,0,0}:Vec{};v.plane_offset=reflection ? d:0;
        v.descriptor=reflection ? Descriptor{1,0,0,d/10}:Descriptor::zero(12);
        if(!reflection) {for(int k=0;k<9;++k) v.descriptor[k]=v.rotation[k]/2;v.descriptor[9]=d/10;}
        votes.push_back(v);
      }
      for(int iterations:{1,100}) {
        Options o;o.bandwidth=.03;o.max_clusters=4;o.max_iterations=iterations;
        std::mt19937_64 a(42),b(42),repeat_rng(42);
        auto x=cpu->clustering(votes.as_view(),{10,{}},o,a);
        auto y=gpu->clustering(votes.as_view(),{10,{}},o,b);
        auto z=gpu->clustering(votes.as_view(),{10,{}},o,repeat_rng);
        need(a==b && b==repeat_rng,"clustering RNG differs");
        need(x.limited_iterations==y.limited_iterations,"iteration diagnostics differ");
        need(x.clusters.size()==y.clusters.size(),"cluster count differs");
        need(x.clusters.votes.offsets==y.clusters.votes.offsets && x.clusters.votes.indices==y.clusters.votes.indices,"membership differs");
        need(y.clusters.votes.indices==z.clusters.votes.indices,"repeat membership differs");
        for(Offset j=0;j<x.clusters.size();++j) {
          auto c=x.clusters[j],d=y.clusters[j],e=z.clusters[j];
          need(c.reflection==d.reflection,"cluster type differs");
          for(int k=0;k<9;++k) {close(c.rotation[k],d.rotation[k]);need(d.rotation[k]==e.rotation[k],"repeat rotation differs");}
          for(int k=0;k<3;++k) {close(c.translation[k],d.translation[k]);close(c.plane_normal[k],d.plane_normal[k]);}
          close(c.plane_offset,d.plane_offset);
        }
      }
    }
    std::cout<<"CUDA clustering passed: mixed types, memberships, centers, RNG, iterations, empty input, repeatability\n";
    return 0;
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
