# One-command publication

Prepared destinations:

- Code: https://github.com/JesseZZZZZ/SpatialScout
- Dataset: https://huggingface.co/datasets/jessezhaoxizhang/SpatialScout

## Upload the code

From the prepared code repository root:

```bash
python3 scripts/upload_github.py
```

Enter a **GitHub personal access token** at the hidden prompt. This is a GitHub
credential, not your model API key. The token needs repository Contents read/write
permission for `JesseZZZZZ/SpatialScout`. Python 3 and `git` are required.
The script clones the current branch into a temporary directory, copies only the
files listed in the release manifest, commits and pushes without force. Obsolete
files from a prior release manifest and retired project plotting scripts are
removed. Other existing remote files are preserved. It does not change the local
working repository or save the token in a remote URL or credential store.

## Upload the dataset

```bash
python3 scripts/upload_huggingface.py --folder /absolute/path/to/SpatialScout
```

Point to the **dataset** folder containing `assets/`, `runtime/`, `benchmark/` and
`MANIFEST.json`. Enter a Hugging Face token with dataset-repository write access.
The script creates `jessezhaoxizhang/SpatialScout` if needed and uploads only the
scene resources, reference task annotations and dataset documentation. No model
outputs or API keys are included. Existing unrelated remote files are preserved.

If necessary, it automatically installs the official `huggingface_hub` client in
`~/.cache/spatialscout-upload` (or `$XDG_CACHE_HOME/spatialscout-upload`). This
requires Python venv support and internet. It does not modify the evaluation venv.
Interrupted uploads can be resumed by re-running the same command.

The parent `release/` directory also contains copies named `upload_github.py`
and `upload_huggingface.py`; the GitHub helper uses the organized clone when available and the HF helper
locates the prepared dataset folder. Noninteractive execution may use `GITHUB_TOKEN` / `HF_TOKEN`; tokens
entered at prompts are not echoed or saved.

Official references:
- https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
- https://huggingface.co/docs/huggingface_hub/guides/upload
