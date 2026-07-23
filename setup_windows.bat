@echo off
REM ===========================================================================
REM  Photo Dedup - one-shot Windows environment setup
REM  Target: Windows 11 + NVIDIA RTX 5070 Super + CUDA 12.x + Python 3.11+
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
where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py -3.11"
) else (
    set "PY=python"
)
%PY% --version
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.11+ not found. Install from https://www.python.org/downloads/
    echo         Make sure "Add python.exe to PATH" is checked.
    exit /b 1
)

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
echo === [4/5] Installing PyTorch (CUDA 12.4) ===
REM If you have no NVIDIA GPU, replace cu124 with "cpu".
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
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
echo    1. Edit config.yaml  ->  set paths.root to your library (e.g. E:/Photos)
echo    2. .venv\Scripts\activate
echo    3. python -m src.stage0_inventory
echo    4. python -m src.stage1_features
echo    5. python -m src.stage2_cluster
echo    6. python -m src.stage3_report
echo    7. open output\review.html in a browser, then run execute_local
echo ===========================================================================
endlocal
