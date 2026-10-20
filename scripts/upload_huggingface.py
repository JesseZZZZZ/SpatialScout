#!/usr/bin/env python3
"""Upload the SpatialScout scene dataset after a hidden Hugging Face token prompt.

Installs the official uploader into a separate cache venv if needed. No login
cache is written. Re-run after an interrupted transfer to reuse uploaded data.
"""
import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys

REPO='jessezhaoxizhang/SpatialScout'
HERE=Path(__file__).resolve().parent
if HERE.name == 'scripts':
    candidates = [HERE.parent.parent/'release/huggingface/SpatialScout',
                  HERE.parent.parents[1]/'huggingface/SpatialScout']
    DEFAULT = next((p for p in candidates if p.is_dir()), HERE.parent.parent/'SpatialScout-data')
else:
    DEFAULT = HERE/'huggingface/SpatialScout' 
PATTERNS=['assets/**','runtime/**','benchmark/**','README.md','THIRD_PARTY_NOTICES.md','MANISKILL_LICENSE','MANIFEST.json','.gitattributes']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=Path(os.environ.get('SPATIALSCOUT_DATA',str(DEFAULT))))
    args=p.parse_args();folder=args.folder.expanduser().resolve()
    for name in ['assets','runtime','benchmark/prepared_80_index.json','MANIFEST.json']:
        if not (folder/name).exists():p.error(f'Missing dataset resource: {folder/name}')
    try:
        from huggingface_hub import HfApi
        from importlib.metadata import version
        if tuple(int(x) for x in version('huggingface-hub').split('.')[:2])<(2,2):raise ImportError
    except (ImportError,ValueError):
        cache=Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'spatialscout-upload'
        python=cache/'bin/python'
        if not python.is_file():
            print('Creating a separate upload environment ...',flush=True)
            subprocess.run([sys.executable,'-m','venv',str(cache)],check=True)
        subprocess.run([str(python),'-m','pip','install','--upgrade','huggingface_hub>=2.2,<3'],check=True)
        os.execv(str(python),[str(python),str(Path(__file__).resolve()),'--folder',str(folder)])
    token=os.environ.get('HF_TOKEN') or getpass.getpass('Hugging Face write token (hidden): ').strip()
    if not token:p.error('A Hugging Face token is required.')
    try:
        api=HfApi(endpoint='https://huggingface.co',token=token)
        api.whoami()
        api.create_repo(repo_id=REPO,repo_type='dataset',exist_ok=True)
        print(f'Uploading scene resources to https://huggingface.co/datasets/{REPO}',flush=True)
        api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(folder),
                          allow_patterns=PATTERNS,
                          ignore_patterns=['**/__pycache__/**','**/.cache/**','**/.git/**','**/*.pyc'],
                          commit_message='Publish SpatialScout scene resources and benchmark annotations')
        print(f'Uploaded: https://huggingface.co/datasets/{REPO}')
    except Exception as exc:
        raise SystemExit('Hugging Face upload failed: '+str(exc).replace(token,'[REDACTED]')) from None


if __name__=='__main__':main()
