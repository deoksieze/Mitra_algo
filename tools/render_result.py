#!/usr/bin/env python3
"""Embed a CPU result in an offline canvas viewer. Open viewer.html in any browser."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('result', type=Path, help='result.json or its output directory')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
source = args.result / 'result.json' if args.result.is_dir() else args.result
result = json.loads(source.read_text())
if result.get('schema_version') != 1:
    parser.error('Unsupported result schema')
# Escaping < prevents user-provided JSON from terminating the inline script.
payload = json.dumps(result, separators=(',',':'), allow_nan=False).replace('<','\\u003c')
template = Path(__file__).with_name('viewer.html').read_text()
target = args.output or source.with_name('viewer.html')
target.write_text(template.replace('/*RESULT_DATA*/null', payload))
print(target.resolve())
