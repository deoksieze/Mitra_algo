#!/usr/bin/env python3
"""Repeat complete CPU/CUDA CLI runs and save reproducible timing evidence."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,default=Path('build-cuda/symmetry_cpu_cli'))
p.add_argument('--repeat',type=int,default=3)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.repeat<1:p.error('repeat must be positive')
root=Path(__file__).resolve().parents[1];records=[]
with tempfile.TemporaryDirectory(prefix='symmetry-suite-') as directory:
    for name in ['reflection','rigid','partial','single']:
        for backend in ['cpu','cuda']:
            runs=[]
            for iteration in range(a.repeat+1):
                out=Path(directory)/f'{name}-{backend}-{iteration}'
                start=time.perf_counter()
                subprocess.run([str(a.binary.resolve()),str(root/'examples'/f'{name}.off'),
                    '--backend',backend,'--output',str(out),'--sample-ratio','1',
                    '--pair-threshold','.08','--verify-threshold','.01'],
                    check=True,capture_output=True,text=True,timeout=180)
                elapsed=(time.perf_counter()-start)*1000
                d=json.loads((out/'result.json').read_text())
                if iteration:runs.append({'wall_ms':elapsed,'timings_ms':d['timings_ms']})
                print(name,backend,'warmup' if not iteration else f'repeat {iteration}',flush=True)
            records.append({'model':name,'backend':backend,'counts':d['counts'],
                'sample':len(d['sample']),'pruned':len(d['pruned']),
                'diagnostics':d.get('diagnostics',{}),'wall_median_ms':statistics.median(r['wall_ms'] for r in runs),
                'stage_median_ms':{s:statistics.median(r['timings_ms'][s] for r in runs) for s in d['timings_ms']},'runs':runs})
metadata={'platform':platform.platform(),'repeat':a.repeat,'warmup':1,
          'parameters':{'sample_ratio':1,'pair_threshold':.08,'verify_threshold':.01,'seed':42},'records':records}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(metadata,indent=2)+'\n')
print(a.output)
