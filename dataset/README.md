# Dataset (not in Git)

The recorded driving data is too large for GitHub (~500MB+).

Download from Google Drive:

https://drive.google.com/file/d/18eGWZ25Gu00CrdBEzYPy1bDvhWF2gn0L/view?usp=share_link

Or from this repo folder:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\download_dataset.py
```

After extraction you should have:

```text
dataset/
├── driving_log.csv
└── IMG/
    └── center_*.jpg
```
