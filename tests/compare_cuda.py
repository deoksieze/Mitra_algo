#!/usr/bin/env python3
"""Compare exported CPU/GPU intermediates and final clusters on synthetic meshes."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
binary=Path(sys.argv[1]).resolve()
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('make_demo',root/'tools/make_demo.py')
generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
maximum=0.0

def compare(a,b,path=''):
    global maximum
    if isinstance(a,float):
        error=abs(a-b)/(1+abs(a));maximum=max(maximum,error)
        assert math.isfinite(b) and error<2e-10,(path,a,b,error)
    elif isinstance(a,list):
        assert len(a)==len(b),(path,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]')
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:compare(a[k],b[k],f'{path}.{k}')
    else:assert a==b,(path,a,b)

with tempfile.TemporaryDirectory(prefix='symmetry-cpu-cuda-') as directory:
    tmp=Path(directory)
    for name in ['reflection','rigid','partial','single']:
        vertices,faces=generator.demo(name)
        model=tmp/f'{name}.off';generator.write_off(model,vertices,faces)
        for ratio in [1,.5]:
            outputs=[]
            for backend in ['cpu','cuda']:
                out=tmp/f'{name}-{ratio}-{backend}'
                subprocess.run([str(binary),str(model),'--backend',backend,'--output',str(out),
                    '--sample-ratio',str(ratio),'--pair-threshold','.08','--verify-threshold','.01'],
                    check=True,capture_output=True,text=True,timeout=60)
                outputs.append(out)
            a,b=outputs
            for filename in ['features.csv','pairs.csv','mesh.off']:
                assert (a/filename).read_bytes()==(b/filename).read_bytes(),filename
            for filename in ['votes.csv','descriptors.csv']:
                rows=[]
                for out in outputs:
                    with (out/filename).open() as f:rows.append(list(csv.reader(f)))
                assert rows[0][0]==rows[1][0] and len(rows[0])==len(rows[1]),filename
                for i,(x,y) in enumerate(zip(rows[0][1:],rows[1][1:])):
                    assert len(x)==len(y)
                    for j,(v,w) in enumerate(zip(x,y)):
                        try:compare(float(v),float(w),f'{filename}[{i},{j}]')
                        except ValueError:assert v==w
            data=[json.loads((out/'result.json').read_text()) for out in outputs]
            for d in data:d.pop('timings_ms')
            compare(*data,path=f'{name}-{ratio}')
    print(f'8 CPU/GPU comparisons passed; max normalized numeric error={maximum:.3g}')
