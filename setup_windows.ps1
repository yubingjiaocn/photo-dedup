# Photo Dedup - native PowerShell environment setup
# Windows 11 + RTX 5070 Ti (Blackwell sm_120) + PyTorch CUDA 12.8

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Step([string]$Text) {
    Write-Host "`n=== $Text ===" -ForegroundColor Cyan
}

function Test-PythonCandidate([string]$Exe, [string[]]$Prefix) {
    try {
        & $Exe @Prefix -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Invoke-SelectedPython([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments) {
    & $script:PythonExe @script:PythonPrefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Write-Step "[1/5] Checking Python"
$script:PythonExe = $null
$script:PythonPrefix = @()

# Prefer the normal PATH command. This works with both the traditional installer
# and the newer Python install manager when its aliases are enabled.
if (Test-PythonCandidate "python" @()) {
    $script:PythonExe = "python"
}
else {
    foreach ($Version in @("3.14", "3.13", "3.12", "3.11")) {
        if (Test-PythonCandidate "py" @("-$Version")) {
            $script:PythonExe = "py"
            $script:PythonPrefix = @("-$Version")
            break
        }
    }
}

if (-not $script:PythonExe) {
    throw @"
No Python 3.11+ runtime was found.
Install/update Python Install Manager, reopen PowerShell, then run this script again.
Recommended fallback: winget install -e --id Python.Python.3.12
"@
}

$VersionText = (& $script:PythonExe @script:PythonPrefix --version 2>&1 | Out-String).Trim()
Write-Host "Using: $script:PythonExe $($script:PythonPrefix -join ' ')"
Write-Host $VersionText
if ($VersionText -match "Python 3\.14") {
    Write-Warning "Trying Python 3.14 as requested. If a dependency lacks a wheel, setup will stop on the exact package instead of silently changing runtimes."
}

Write-Step "[2/5] Creating virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Invoke-SelectedPython @("-m", "venv", ".venv")
}
else {
    Write-Host ".venv already exists; reusing it."
}
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Step "[3/5] Upgrading pip"
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }

Write-Step "[4/5] Installing PyTorch CUDA 12.8"
& $VenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch cu128 installation failed. CPU fallback is intentionally disabled because this machine should use the RTX 5070 Ti."
}

Write-Step "[5/5] Installing project requirements"
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed" }

Write-Step "Verifying GPU visibility"
& $VenvPython -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if ($LASTEXITCODE -ne 0) { throw "Torch verification failed" }

$CudaVisible = & $VenvPython -c "import torch; print('yes' if torch.cuda.is_available() else 'no')"
if (($CudaVisible | Select-Object -Last 1).Trim() -ne "yes") {
    throw "PyTorch installed, but CUDA is not available. Update the NVIDIA driver and send the console output."
}

Write-Host "`n===========================================================================" -ForegroundColor Green
Write-Host "Setup complete. No photo was moved, modified, or deleted."
Write-Host "Next:"
Write-Host '  .\.venv\Scripts\python.exe -m src.run_pipeline --root "E:\Photos" --output "C:\photo-review" --backend torch --limit 100 --no-serve'
Write-Host 'If it fails, send C:\photo-review\photo-dedup.log'
Write-Host "===========================================================================" -ForegroundColor Green
