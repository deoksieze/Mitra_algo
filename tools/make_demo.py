#!/usr/bin/env python3
"""Generate small consistently wound meshes; no third-party Python dependencies."""
import argparse
import math
from pathlib import Path


def shape(level=2):
    vertices = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    faces = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    for _ in range(level):
        mids, new_faces = {}, []
        def midpoint(a,b):
            key = tuple(sorted((a,b)))
            if key not in mids:
                p = tuple(vertices[a][k]+vertices[b][k] for k in range(3))
                length = math.sqrt(sum(x*x for x in p))
                mids[key] = len(vertices)
                vertices.append(tuple(x/length for x in p))
            return mids[key]
        for a,b,c in faces:
            ab,bc,ca = midpoint(a,b),midpoint(b,c),midpoint(c,a)
            new_faces += [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]
        faces = new_faces
    # Distinct principal curvatures and no deliberately imposed internal mirror.
    vertices = [(0.7*x+0.11*y*z+0.09*z*z,
                 0.48*y+0.07*x*x+0.05*z*x,
                 z+0.12*x*y+0.08*x) for x,y,z in vertices]
    return vertices, faces


def demo(kind='reflection', level=2):
    vertices, faces = shape(level)
    first = [(x-1.6,y,z) for x,y,z in vertices]
    if kind == 'reflection':
        second = [(-x,y,z) for x,y,z in first]
        other_faces = [(a,c,b) for a,b,c in faces]
    elif kind == 'rigid':
        angle = 0.65
        c,s = math.cos(angle), math.sin(angle)
        second = [(c*x-s*y+3.2,s*x+c*y+0.3,z+0.2) for x,y,z in first]
        other_faces = faces
    elif kind == 'partial':
        second = [(-x,y,z) for x,y,z in first]
        n=len(first)
        both=first+second
        fs=faces+[(a+n,c+n,b+n) for a,b,c in faces]
        n=len(both)
        decoy=[(1.3*x+.4,1.3*y+4,1.3*z+.5) for x,y,z in vertices]
        return both+decoy, fs+[(a+n,b+n,c+n) for a,b,c in faces]
    elif kind == 'single':
        return first, faces
    else:
        raise ValueError(kind)
    n = len(first)
    return first+second, faces+[(a+n,b+n,c+n) for a,b,c in other_faces]


def write_off(path, vertices, faces):
    with Path(path).open('w') as f:
        f.write(f'OFF\n{len(vertices)} {len(faces)} 0\n')
        for p in vertices:
            f.write(' '.join(format(x,'.17g') for x in p)+'\n')
        for face in faces:
            f.write('3 '+' '.join(map(str,face))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('examples'))
    parser.add_argument('--level', type=int, choices=range(4), default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for kind in ['reflection', 'rigid', 'partial', 'single']:
        path = args.output / f'{kind}.off'
        write_off(path, *demo(kind,args.level))
        print(path)
