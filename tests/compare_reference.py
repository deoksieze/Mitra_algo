#!/usr/bin/env python3
"""Compare two CLI builds on identical fixtures (no binaries stored in Git)."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference',type=Path)
parser.add_argument('candidate',type=Path)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
args=parser.parse_args()
with tempfile.TemporaryDirectory(prefix='symmetry-reference-') as directory:
    for scenario in ['reflection','rigid','partial','single']:
        for sample in [1,.5]:
            outputs=[]
            for i,binary in enumerate([args.reference,args.candidate]):
                out=Path(directory)/f'{scenario}-{sample}-{i}'
                subprocess.run([str(binary.resolve()),str(args.root/'examples'/f'{scenario}.off'),
                                '--output',str(out),'--sample-ratio',str(sample),'--pair-threshold','.08',
                                '--verify-threshold','.01'],check=True,capture_output=True,text=True,timeout=120)
                outputs.append(out)
            old=json.loads((outputs[0]/'result.json').read_text())
            new=json.loads((outputs[1]/'result.json').read_text())
            old.pop('timings_ms');new.pop('timings_ms')
            assert old==new,f'{scenario}, sample {sample}: JSON mismatch'
            for filename in ['features.csv','pairs.csv','votes.csv','descriptors.csv','mesh.off']:
                assert (outputs[0]/filename).read_bytes()==(outputs[1]/filename).read_bytes(),filename
            print(f'{scenario}, sample {sample}: exact match')
