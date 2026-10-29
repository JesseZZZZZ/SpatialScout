# Publication helpers

Review materials contain no predefined publication destination. To publish a
separate copy, provide your own destination explicitly:

```bash
python3 scripts/upload_github.py --remote "$CODE_REMOTE"
python3 scripts/upload_huggingface.py --repo-id "$DATASET_REPO_ID" --folder "$SPATIALSCOUT_DATA"
```

Each helper prompts for a platform token, or reads `GITHUB_TOKEN` / `HF_TOKEN`.
The Git helper uses an anonymous commit identity. Publishing under a personal
account can reveal its owner; distribute the review archive through the
submission system. Do not include the working checkout's `.git` directory.

## Push an anonymous review branch

Log in once with `gh auth login`, then run from the release directory:

```bash
# Optional: verify the snapshot without any network requests.
python3 scripts/push_anonymous_branch.py --repo OWNER/REPO --dry-run
python3 scripts/push_anonymous_branch.py --repo OWNER/REPO --branch anonymous-review
```

This helper verifies the file sizes and hashes in `MANIFEST.json`, creates a
separate temporary repository, and pushes exactly one root commit. Its author
and committer are `Anonymous`, and the commit dates are fixed. No original
commits, tags, remotes, hooks or local Git settings are copied. The working
checkout and its current branch are unchanged. An existing remote branch is
rejected; choose a new branch name to publish another snapshot. If release
files change, update their inventory entries before using the helper.

GitHub can associate the push with the logged-in account. Other branches in the
destination repository can also reveal author identities. For anonymous review,
use a separate repository and an account whose public information is anonymous.
