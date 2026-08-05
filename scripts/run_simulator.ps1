# Start the inference server, then open the Udacity simulator in Autonomous Mode.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Missing .venv. Create it first: python -m venv .venv"
}

.\.venv\Scripts\Activate.ps1

$TargetSpeed = if ($args.Count -ge 1) { $args[0] } else { "9" }
$MinimumSpeed = if ($args.Count -ge 2) { $args[1] } else { "6" }

Write-Host "Starting TestSimulation.py (target=$TargetSpeed, min=$MinimumSpeed)"
Write-Host "Leave this window open, then launch the simulator and choose Autonomous Mode."

python .\TestSimulation.py --model .\model.h5 --target-speed $TargetSpeed --minimum-speed $MinimumSpeed
