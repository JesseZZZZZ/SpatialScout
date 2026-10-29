#!/usr/bin/env python3
"""Upload the prepared SpatialScout code repository after a hidden PAT prompt.

Requires Python 3 and git. Never writes the token into the remote URL or Git config.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent if HERE.name == 'scripts' else HERE/'github/SpatialScout'
REMOVED = {'analyze_astra_deepseek_grounding.py','benchmark_sim_parallelism.py','evaluate_saved_results.py',
           'export_current_evaluation.py','export_render_bundle.py','make_qualitative_figure.py',
           'prepare_reproduction_release.py','requirements-analysis.txt'}


def relative_file(name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts or '.git' in path.parts:
        raise ValueError(f'Invalid manifest path: {name}')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', required=True, help='Destination Git repository supplied by the user')
    parser.add_argument('--folder', type=Path, default=DEFAULT)
    args=parser.parse_args()
    source=args.folder.expanduser().resolve()
    remote=args.remote
    if not shutil.which('git'):parser.error('git is required; install it before uploading.')
    if not (source/'MANIFEST.json').is_file():parser.error(f'MANIFEST.json not found in {source}')
    manifest=json.loads((source/'MANIFEST.json').read_text())
    files=[relative_file(row['path']) for row in manifest['files']]+[Path('MANIFEST.json')]
    for name in files:
        p=source/name
        if not p.is_file() or p.is_symlink():parser.error(f'Missing or symlinked release file: {name}')
    token=os.environ.get('GITHUB_TOKEN') or getpass.getpass('GitHub personal access token (hidden): ').strip()
    if not token:parser.error('A GitHub token is required.')
    try:
        with tempfile.TemporaryDirectory(prefix='spatialscout-github-') as tmp:
            temporary=Path(tmp);checkout=temporary/'repo';helper=temporary/'askpass.py'
            helper.write_text('#!'+sys.executable+'\nimport os,sys\nprompt=" ".join(sys.argv[1:]).lower()\nprint("x-access-token" if "username" in prompt else os.environ["SPATIALSCOUT_UPLOAD_TOKEN"])\n')
            helper.chmod(0o700)
            env=os.environ.copy()
            for key in ['GIT_TRACE','GIT_TRACE_CURL','GIT_CURL_VERBOSE','GIT_TRACE_PACKET']:
                env.pop(key,None)
            env.update(GIT_ASKPASS=str(helper),GIT_TERMINAL_PROMPT='0',SPATIALSCOUT_UPLOAD_TOKEN=token)
            def git(*arguments,cwd=None):
                result=subprocess.run(['git','-c','credential.helper=',*arguments],cwd=cwd,env=env,text=True,capture_output=True)
                if result.returncode:
                    raise RuntimeError((result.stderr or result.stdout).strip().replace(token,'[REDACTED]'))
                return result.stdout.strip()
            print(f'Uploading {len(files)} release files to {remote}',flush=True)
            git('clone','--depth','1',remote,str(checkout))
            branch=git('branch','--show-current',cwd=checkout) or 'main'
            if not (checkout/'.git/refs/heads'/branch).exists():
                # For a completely empty remote, use main for the first commit.
                status=subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=checkout,capture_output=True)
                if status.returncode:git('symbolic-ref','HEAD','refs/heads/main',cwd=checkout);branch='main'
            # Remove only obsolete files from a previous SpatialScout release manifest.
            old_manifest=checkout/'MANIFEST.json'
            new_names={str(name) for name in files}
            if old_manifest.is_file():
                for row in json.loads(old_manifest.read_text()).get('files',[]):
                    name=relative_file(row['path']);old=checkout/name
                    if str(name) not in new_names and old.is_file() and not old.is_symlink():old.unlink()
            # Also remove the specifically retired project drawing/analysis files.
            for p in list(checkout.glob('*.py')):
                if p.name.startswith('plot_') or p.name in REMOVED:p.unlink()
            for name in ['paper','release_templates']:
                p=checkout/name
                if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
            for name in files:
                destination=checkout/name;destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.is_symlink():destination.unlink()
                shutil.copy2(source/name,destination)
            git('config','user.name','Anonymous',cwd=checkout)
            git('config','user.email','anonymous',cwd=checkout)
            git('add','--all',cwd=checkout)
            if not git('status','--porcelain',cwd=checkout):
                print('The remote release already matches these files.');return
            git('commit','-m','Publish SpatialScout evaluation code and documentation',cwd=checkout)
            git('push','origin',f'HEAD:{branch}',cwd=checkout)
        print('Upload complete.')
    except Exception as exc:
        raise SystemExit('GitHub upload failed: '+str(exc).replace(token,'[REDACTED]')) from None


if __name__=='__main__':main()
