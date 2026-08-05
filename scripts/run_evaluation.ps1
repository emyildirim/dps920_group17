# Offline evaluation helper for Group 17.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Missing .venv. Create it first: python -m venv .venv"
}

.\.venv\Scripts\Activate.ps1

$Csv = Join-Path $Root "dataset\driving_log.csv"
if (-not (Test-Path $Csv)) {
    Write-Host "Dataset missing. Downloading..."
    python .\scripts\download_dataset.py
}

python .\evaluation.py `
  --model .\model.h5 `
  --csv .\dataset\driving_log.csv `
  --image-root .\dataset `
  --output-dir .\results

Write-Host "Done. Open results\metrics.json and the PNG plots."
