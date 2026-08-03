@echo off
REM ===========================================================================
REM  Photo Dedup - one-shot Windows environment setup
REM  Target: Windows 11 + NVIDIA RTX 5070 Ti + Python 3.12 + PyTorch cu128
REM ===========================================================================
REM  What this does:
REM    1. Creates a local virtual environment in .venv
REM    2. Installs PyTorch CUDA 12.4 wheels (torch + torchvision)
REM    3. Installs the rest of requirements.txt
REM    4. Verifies torch sees the GPU
REM  Re-runnable: safe to run again; it reuses the existing .venv.
REM ===========================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo === [1/5] Checking Python ===
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3.11 --version >nul 2>&1 && set "PY=py -3.11"
if not defined PY (
    python -c "import sys; raise SystemExit(0 if sys.version_info ^>= (3,11) else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY=python"
)
if not defined PY (
    echo [ERROR] A supported Python was not found.
    echo         Python 3.12 is recommended.
    echo.
    echo         Install it with:
    echo           winget install -e --id Python.Python.3.12
    echo         Then close and reopen PowerShell and run this script again.
    exit /b 1
)
echo Using: %PY%
%PY% --version
%PY% -c "import sys; print('[WARN] Python 3.14 is being tried as requested; if a dependency has no wheel, install 3.12 side-by-side.') if sys.version_info ^>= (3,14) else None"

echo.
echo === [2/5] Creating virtual environment (.venv) ===
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if %errorlevel% neq 0 ( echo [ERROR] venv creation failed & exit /b 1 )
) else (
    echo .venv already exists, reusing.
)
call ".venv\Scripts\activate.bat"

echo.
echo === [3/5] Upgrading pip ===
python -m pip install --upgrade pip setuptools wheel

echo.
echo === [4/5] Installing PyTorch (CUDA 12.8 / RTX 50 series) ===
REM CUDA 12.8 wheels include Blackwell sm_120 support for the RTX 5070 Ti.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo [WARN] CUDA torch install failed. Falling back to CPU wheels.
    python -m pip install torch torchvision
)

echo.
echo === [5/5] Installing project requirements ===
python -m pip install -r requirements.txt
if %errorlevel% neq 0 ( echo [ERROR] requirements install failed & exit /b 1 )

echo.
echo === Verifying GPU visibility ===
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo.
echo ===========================================================================
echo  Setup complete.
echo  Next:
echo    1. .venv\Scripts\activate
echo    2. python -m src.run_pipeline --root "E:\Photos" --output "C:\photo-review" --backend torch --limit 100 --no-serve
echo    3. If anything fails, send C:\photo-review\photo-dedup.log
echo.
echo  This setup and the command above never move or delete photos.
echo ===========================================================================
endlocal
