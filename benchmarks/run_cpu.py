#!/usr/bin/env python3
"""Report per-stage medians from repeat CLI runs, without persistent results."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model',type=Path)
p.add_argument('--binary',type=Path,default=Path('build/symmetry_cpu_cli'))
p.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
p.add_argument('--repeat',type=int,default=3)
p.add_argument('--sample-ratio',type=float,default=.5)
a=p.parse_args()
if a.repeat<1:p.error('--repeat must be positive')
values={}
with tempfile.TemporaryDirectory(prefix='symmetry-benchmark-') as directory:
    for i in range(a.repeat+1):
        out=Path(directory)/str(i)
        subprocess.run([str(a.binary.resolve()),str(a.model.resolve()),'--output',str(out),
                        '--backend',a.backend,'--sample-ratio',str(a.sample_ratio)],check=True,capture_output=True,text=True)
        data=json.loads((out/'result.json').read_text())
        if i:
            for stage,ms in data['timings_ms'].items():values.setdefault(stage,[]).append(ms)
    print('counts:',data['counts'])
    print('stage,median_ms')
    for stage,ms in values.items():print(f'{stage},{statistics.median(ms):.6f}')
