"""Attach downloaded scene resources without copying model outputs or credentials."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('data_dir',type=Path)
p.add_argument('--verify',action='store_true',help='Verify all dataset files using SHA-256 before attaching')
a=p.parse_args(); data=a.data_dir.expanduser().resolve()
if not (data/'benchmark/prepared_80_index.json').is_file():p.error('Expected a SpatialScout dataset directory containing benchmark/prepared_80_index.json')
if a.verify:
    manifest=json.loads((data/'MANIFEST.json').read_text())
    for row in manifest['files']:
        f=data/row['path'];h=hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=row['sha256']:raise SystemExit(f'Checksum mismatch: {row["path"]}')
def link(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink():
        if destination.resolve()==source.resolve():return
        raise SystemExit(f'Existing link points elsewhere: {destination}')
    if destination.exists():raise SystemExit(f'Path exists; move it aside first: {destination}')
    destination.symlink_to(source,target_is_directory=source.is_dir())
link(data/'assets',ROOT/'assets')
link(data/'benchmark',ROOT/'benchmark_data')
for f in sorted((data/'runtime').rglob('*')):
    if f.is_file():link(f,ROOT/f.relative_to(data/'runtime'))
print(f'Attached dataset: {data}')
print('Keep this data directory available; the code uses symlinks to it.')
