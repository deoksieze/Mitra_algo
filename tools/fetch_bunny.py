#!/usr/bin/env python3
"""Fetch Stanford's coarse bunny reconstruction and convert its ASCII PLY to OFF."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
if a.output.exists():p.error('output already exists')
url='https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
try:archive=urllib.request.urlopen(url,timeout=60).read()
except OSError as error:p.error(f'Cannot fetch Stanford archive: {error}')
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
    name=next(m.name for m in tar.getmembers() if m.name.endswith('/bun_zipper_res4.ply'))
    text=tar.extractfile(name).read()
lines=text.decode('ascii').splitlines();assert lines[0]=='ply' and lines[1]=='format ascii 1.0'
end=lines.index('end_header');nv=nf=0
for line in lines[:end]:
    if line.startswith('element vertex '):nv=int(line.split()[-1])
    if line.startswith('element face '):nf=int(line.split()[-1])
vertices=[line.split()[:3] for line in lines[end+1:end+1+nv]]
faces=[line.split() for line in lines[end+1+nv:end+1+nv+nf]]
assert nv and nf and all(int(f[0])==3 and len(f)==4 for f in faces)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w') as f:
    f.write(f'OFF\n{nv} {nf} 0\n')
    for row in vertices+faces:f.write(' '.join(row)+'\n')
metadata={'source':url,'member':name,'archive_sha256':hashlib.sha256(archive).hexdigest(),
          'ply_sha256':hashlib.sha256(text).hexdigest(),'vertices':nv,'faces':nf}
a.output.with_suffix('.source.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
