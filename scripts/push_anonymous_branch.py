#!/usr/bin/env python3
"""Push a verified release snapshot as a new branch with one anonymous commit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def git_env():
    # Ignore inherited Git identities, repository paths and configuration overrides.
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update(
        GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT='0',
        GIT_AUTHOR_NAME='Anonymous', GIT_AUTHOR_EMAIL='anonymous',
        GIT_COMMITTER_NAME='Anonymous', GIT_COMMITTER_EMAIL='anonymous',
        GIT_AUTHOR_DATE='2000-01-01T00:00:00+0000',
        GIT_COMMITTER_DATE='2000-01-01T00:00:00+0000',
    )
    return env


def git(*args, cwd, env, check=True):
    return subprocess.run(
        ['git', '-c', 'core.hooksPath=' + os.devnull,
         '-c', 'commit.gpgsign=false', '-c', 'credential.helper=',
         '-c', 'credential.helper=!gh auth git-credential', *args],
        cwd=cwd, env=env, text=True, capture_output=True, check=check,
    )


def copy_release(source, checkout):
    """Copy only the manifest inventory; never traverse a symlink or copy Git data."""
    manifest_path = source / 'MANIFEST.json'
    if manifest_path.is_symlink():
        raise ValueError('MANIFEST.json must be a regular file')
    manifest_bytes = manifest_path.read_bytes()
    rows = json.loads(manifest_bytes)['files']
    seen = {'MANIFEST.json'}
    for row in rows:
        name = row['path']
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or '..' in path.parts or
                '.git' in path.parts or '\\' in name or str(path) != name or name in seen):
            raise ValueError(f'Invalid or duplicate release path: {name}')
        seen.add(name)
        original = source.joinpath(*path.parts)
        current = source
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f'Symlinked release path: {name}')
        if not original.is_file():
            raise ValueError(f'Missing release file: {name}')
        data = original.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Manifest mismatch: {name}; refresh MANIFEST.json before publishing')
        destination = checkout.joinpath(*path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (checkout / 'MANIFEST.json').write_bytes(manifest_bytes)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='Destination repository: OWNER/REPO')
    parser.add_argument('--branch', default='anonymous-review', help='New branch name (default: anonymous-review)')
    parser.add_argument('--folder', type=Path, default=ROOT, help='Release directory containing MANIFEST.json')
    parser.add_argument('--dry-run', action='store_true', help='Verify files and build the anonymous commit locally, without network access')
    args = parser.parse_args()
    if not shutil.which('git'):
        parser.error('git is required')
    if args.repo.startswith('-') or args.repo.count('/') != 1 or any(c.isspace() for c in args.repo):
        parser.error('--repo must be OWNER/REPO')
    if args.branch in {'main', 'master', 'HEAD'} or args.branch.startswith('-'):
        parser.error('Choose a new review branch name')
    source = args.folder.expanduser().resolve()
    env = git_env()
    try:
        git('check-ref-format', '--branch', args.branch, cwd=source, env=env)
        with tempfile.TemporaryDirectory(prefix='anonymous-release-') as tmp:
            checkout = Path(tmp)
            count = copy_release(source, checkout)
            git('init', '--template=', '--initial-branch=' + args.branch, cwd=checkout, env=env)
            git('add', '--force', '--all', cwd=checkout, env=env)
            git('commit', '-m', 'Anonymous review snapshot', cwd=checkout, env=env)
            if git('rev-list', '--count', 'HEAD', cwd=checkout, env=env).stdout.strip() != '1':
                raise ValueError('Snapshot unexpectedly contains Git history')
            print(f'Verified {count} release files; one root commit by Anonymous.', flush=True)
            if args.dry_run:
                print('Dry run complete; no network requests or pushes were made.')
                return
            if not shutil.which('gh'):
                raise ValueError('GitHub CLI is required; run gh auth login before pushing')
            subprocess.run(['gh', 'auth', 'status'], check=True)
            remote = subprocess.run(
                ['gh', 'repo', 'view', args.repo, '--json', 'url', '--jq', '.url'],
                text=True, capture_output=True, check=True,
            ).stdout.strip()
            if not remote:
                raise ValueError('GitHub CLI returned an empty repository address')
            ref = 'refs/heads/' + args.branch
            existing = git('ls-remote', '--exit-code', '--heads', remote, ref,
                           cwd=checkout, env=env, check=False)
            if existing.returncode == 0:
                raise ValueError('The remote branch already exists; choose another --branch. Nothing was overwritten.')
            if existing.returncode != 2:
                raise ValueError(existing.stderr.strip() or 'Cannot inspect the remote repository')
            git('push', '--no-follow-tags', remote, 'HEAD:' + ref, cwd=checkout, env=env)
            print(f'Pushed {args.repo}, branch {args.branch}.')
            print('Commit metadata is anonymous; GitHub may still identify the account that pushed it.')
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            message = (exc.stderr or '').strip() or f'Command failed with exit code {exc.returncode}'
        else:
            message = str(exc)
        parser.exit(1, message + '\n')


if __name__ == '__main__':
    main()
