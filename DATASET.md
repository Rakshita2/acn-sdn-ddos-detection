FlowStatsfile.csv

This repository previously contained a large dataset file `dataset/FlowStatsfile.csv` (369+ MB) that exceeds GitHub's file size limits. The file has been removed from the repository history and moved to a local dataset directory to keep the repository lightweight.

Location (on this machine)
- ~/datasets/FlowStatsfile.csv

If you don't have the file locally
- Option A (recommended for collaborators): upload the file to a cloud storage (S3, Google Drive, or an internal file server) and download it to `~/datasets` when needed.
- Option B (one-off sharing): use `scp`/`rsync` to copy the file from a machine that has it.

Example commands
- Create the local datasets folder (if missing):
  mkdir -p ~/datasets

- Copy the dataset into the project for a run (temporary copy):
  cp ~/datasets/FlowStatsfile.csv dataset/FlowStatsfile.csv

- Load the dataset in Python (pandas):
  import pandas as pd
  df = pd.read_csv('dataset/FlowStatsfile.csv')

Why this was done
- GitHub rejects files >100MB and keeping large binary files in git inflates history and makes cloning slow.
- The dataset was removed from git history and is now ignored by `.gitignore`.

Notes
- `dataset/FlowStatsfile.csv` is listed in `.gitignore`. Copying it into `dataset/` is fine for local runs but will not be tracked or pushed.
- If you want the dataset to be stored in the repository via Git LFS, we can migrate it to LFS instead — let me know.

Contact
- If you need me to move the file to a different path or upload it to a cloud location, tell me where and I will do it.
