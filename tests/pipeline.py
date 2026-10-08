#!/usr/bin/env python3
"""End-to-end regression against known geometry, including parser/error contracts."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

binary, root = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('make_demo', root/'tools/make_demo.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def run(args, success=True):
    process = subprocess.run([str(binary),*map(str,args)], capture_output=True, text=True, timeout=45)
    if (process.returncode == 0) != success:
        raise AssertionError(process.stdout+process.stderr)
    return process


def finite(value):
    if isinstance(value,float):
        assert math.isfinite(value)
    elif isinstance(value,dict):
        for v in value.values():
            finite(v)
    elif isinstance(value,list):
        for v in value:
            finite(v)


def stable(data):
    data = copy.deepcopy(data)
    data.pop('timings_ms')
    return data


def apply(c,p):
    r,t = c['rotation'],c['translation']
    return [sum(r[i*3+j]*p[j] for j in range(3))+t[i] for i in range(3)]


with tempfile.TemporaryDirectory(prefix='symmetry-regression-') as directory:
    tmp = Path(directory)
    def detect(name,vertices,faces,extra=()):
        model = tmp/f'{name}.off'
        generator.write_off(model,vertices,faces)
        out = tmp/name
        run([model,'--output',out,'--sample-ratio','1','--pair-threshold','.08',
             '--verify-threshold','.01',*extra])
        data = json.loads((out/'result.json').read_text())
        finite(data)
        return data,model,out

    vertices,faces = generator.demo('reflection')
    reflection,model,out = detect('reflection',vertices,faces)
    mirror = reflection['clusters'][0]
    assert mirror['type'] == 'reflection'
    assert abs(abs(mirror['plane_normal'][0])-1)<1e-5
    assert abs(mirror['plane_offset'])<1e-4
    assert mirror['coverage']>.95 and mirror['rms']/reflection['box_size']<1e-4
    assert mirror['side_a'] and mirror['side_b'] and 0 in mirror['side_a']
    assert mirror['patches_a'] and mirror['patches_b']
    repeat,_,repeat_out = detect('repeat',vertices,faces)
    assert stable(reflection) == stable(repeat)
    for filename in ['features.csv','pairs.csv','votes.csv','descriptors.csv','mesh.off']:
        assert (out/filename).read_bytes() == (repeat_out/filename).read_bytes()
    # Existing outputs are protected even if the input is valid.
    process = run([model,'--output',out],False)
    assert 'new or empty' in process.stderr

    vertices_r,faces_r = generator.demo('rigid')
    rigid,_,_ = detect('rigid',vertices_r,faces_r)
    c = rigid['clusters'][0]
    assert c['type']=='rigid' and c['coverage']>.95
    half=len(vertices_r)//2
    errors=[math.dist(apply(c,p),q) for p,q in zip(vertices_r[:half],vertices_r[half:])]
    assert max(errors)<1e-6

    # Change the global coordinate system: plane canonicalization and centered metric.
    angle=.7
    rotated=[(math.cos(angle)*x-math.sin(angle)*y+8,
              math.sin(angle)*x+math.cos(angle)*y-3,z+2) for x,y,z in vertices]
    arbitrary,_,_ = detect('arbitrary_plane',rotated,faces)
    expected=(math.cos(angle),math.sin(angle),0)
    candidates=[c for c in arbitrary['clusters'] if c['type']=='reflection' and
                abs(sum(a*b for a,b in zip(c['plane_normal'],expected)))>.999]
    assert candidates and max(c['coverage'] for c in candidates)>.95

    # A third, scaled unrelated component makes the same symmetry genuinely partial.
    decoy,df=generator.shape()
    n=len(vertices)
    mixed=vertices+[(1.3*x+.4,1.3*y+4,1.3*z+.5) for x,y,z in decoy]
    partial,_,_=detect('partial',mixed,faces+[(a+n,b+n,c+n) for a,b,c in df])
    candidates=[c for c in partial['clusters'] if c['type']=='reflection' and
                abs(c['plane_normal'][0])>.999 and abs(c['plane_offset'])<.03]
    assert candidates
    correct=max(candidates,key=lambda c:len(c['side_a'])+len(c['side_b']))
    assert .35<correct['coverage']<.85
    assert len(correct['side_a'])>=len(vertices)//2*.9

    # Sampling actually varies by seed, while the same seed remains reproducible.
    sampled,_,_=detect('sampled',vertices,faces,['--sample-ratio','.5'])
    sampled_repeat,_,_=detect('sampled_repeat',vertices,faces,['--sample-ratio','.5'])
    sampled_other,_,_=detect('sampled_other',vertices,faces,['--sample-ratio','.5','--seed','7'])
    assert stable(sampled)==stable(sampled_repeat)
    assert sampled['sample']!=sampled_other['sample']
    # Small perturbations should preserve a high-coverage mirror hypothesis.
    noisy=vertices.copy()
    half=len(vertices)//2
    for i in range(half,len(vertices)):
        x,y,z=noisy[i]
        noisy[i]=(x+.001*math.sin(i),y+.001*math.cos(i),z+.001*math.sin(i*.7))
    noise,_,_=detect('noise',noisy,faces)
    assert any(c['type']=='reflection' and abs(c['plane_normal'][0])>.999 and
               c['coverage']>.9 for c in noise['clusters'])

    # Small uncurved patch: no unstable tangent directions, no votes, valid empty result.
    planar,planar_model,_=detect('plane',[(0,0,0),(1,0,0),(1,1,0),(0,1,0)],[(0,1,2),(0,2,3)])
    assert not planar['pruned'] and planar['counts']['votes']==0 and not planar['clusters']
    subdivided,_,_=detect('subdivide',[(0,0,0),(1,0,0),(1,1,0),(0,1,0)],[(0,1,2),(0,2,3)],['--subdivide','1'])
    assert subdivided['counts']['faces']==8 and subdivided['counts']['vertices']==9
    no_patch,_,_=detect('no_patch',vertices,faces,['--no-patching'])
    assert no_patch['clusters'] and all(not c['side_a'] for c in no_patch['clusters'])

    # OBJ polygon fan, negative indices and a mixed-case filename on Linux.
    obj=tmp/'MixedCase.OBJ'
    obj.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4/1 -3/2 -2/3 -1/4\n')
    run([obj,'--output',tmp/'obj'])
    assert json.loads((tmp/'obj/result.json').read_text())['counts']['faces']==2
    bad=tmp/'bad.off';bad.write_text('OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 9\n')
    assert 'range' in run([bad,'--output',tmp/'bad'],False).stderr
    bad.write_text('OFF\n3 1 0\n0 0 0\n1 0 0\n2 0 0\n3 0 1 2\n')
    assert 'nondegenerate' in run([bad,'--output',tmp/'degenerate'],False).stderr
    flipped=faces.copy();a,b,c=flipped[0];flipped[0]=(a,c,b)
    generator.write_off(bad,vertices,flipped)
    assert 'winding' in run([bad,'--output',tmp/'winding'],False).stderr
    for option,value in [('--sample-ratio','0'),('--bandwidth','nan'),('--smoothing','-1'),('--prune','1'),('--seed','-1'),('--max-clusters','1.5')]:
        run([model,'--output',tmp/'invalid',option,value],False)
    assert 'budget' in run([model,'--output',tmp/'budget','--sample-ratio','1','--max-pairs','1'],False).stderr
    assert 'budget' in run([model,'--output',tmp/'vote_budget','--sample-ratio','1','--max-votes','1'],False).stderr
    # The renderer must be standalone and contain actual finite results.
    subprocess.run([sys.executable,str(root/'tools/render_result.py'),str(out)],check=True,capture_output=True)
    html=(out/'viewer.html').read_text()
    assert '/*RESULT_DATA*/null' not in html and '"schema_version":1' in html
    assert 'https://' not in html and 'fetch(' not in html
    print('Pipeline checks passed: reflection, rigid, partial, coordinates, determinism, errors, viewer')
