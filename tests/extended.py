#!/usr/bin/env python3
"""Additional geometry and failure cases, tested on both backends."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
binary=Path(sys.argv[1]).resolve();root=Path(sys.argv[2])
spec=importlib.util.spec_from_file_location('demo',root/'tools/make_demo.py');demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)

def compare(a,b):
    if isinstance(a,float) or isinstance(b,float):assert abs(a-b)<1e-9*(1+abs(a)),(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b,(a,b)

vertices,faces=demo.demo('reflection',level=1)
noise=vertices.copy();half=len(vertices)//2
for i in range(half,len(noise)):
    x,y,z=noise[i];noise[i]=(x+.0001*math.sin(i),y+.0001*math.cos(i),z)
scaled=vertices[:half]+[(1.3*x+.4,1.3*y+3,1.3*z+.7) for x,y,z in vertices[half:]]
sphere=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
sf=[(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
cases={'sphere_octahedron':(sphere,sf),'small_reflection':(vertices,faces),'small_noise':(noise,faces),
       'scaled_copy':(scaled,faces),'single_asymmetric':demo.demo('single',level=1)}
with tempfile.TemporaryDirectory(prefix='symmetry-extended-') as directory:
    tmp=Path(directory)
    for name,(v,f) in cases.items():
        model=tmp/f'{name}.off';demo.write_off(model,v,f);results=[]
        for backend in ['cpu','cuda']:
            out=tmp/f'{name}-{backend}'
            subprocess.run([str(binary),str(model),'--backend',backend,'--output',str(out),'--sample-ratio','1',
                            '--pair-threshold','.08','--verify-threshold','.01'],capture_output=True,text=True,check=True,timeout=90)
            d=json.loads((out/'result.json').read_text());d.pop('timings_ms');results.append(d)
        compare(*results)
        if name=='sphere_octahedron':assert not results[0]['pruned'] and not results[0]['clusters']
        if name=='small_reflection':assert any(c['type']=='reflection' and c['coverage']>.95 for c in results[0]['clusters'])
        assert results[0]['mode']=='rigid_and_plane_reflection'
        print(name,'passed',flush=True)
    for text in ['OFF\n0 0 0\n','OFF\n3 1 0\n0 0 0\n1 0 0\n2 0 0\n3 0 1 2\n']:
        bad=tmp/'bad.off';bad.write_text(text)
        for backend in ['cpu','cuda']:
            p=subprocess.run([str(binary),str(bad),'--backend',backend,'--output',str(tmp/'bad-result')],capture_output=True,text=True,timeout=30)
            assert p.returncode!=0 and not (tmp/'bad-result').exists()
    print('Extended CPU/GPU cases and invalid inputs passed')
