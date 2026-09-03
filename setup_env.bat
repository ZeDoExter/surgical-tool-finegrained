@echo off
REM ============================================================
REM  setup_env.bat - create the training environment on a new PC
REM  Usage: double-click, or run in a terminal:
REM         cd DentalInstrument   (repo root)
REM         setup_env.bat
REM  Needs: Python 3.12 (python.org or `py -3.12`) on PATH
REM ============================================================
setlocal
cd /d "%~dp0"

echo [1/4] Checking Python 3.12 ...
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 --version >nul 2>nul
    if %errorlevel%==0 (
        set "PYCMD=py -3.12"
        goto :create
    )
)
python --version 2>nul | findstr /b "Python 3.12" >nul
if %errorlevel%==0 (
    set "PYCMD=python"
    goto :create
)
echo.
echo   ERROR: Python 3.12 not found.
echo   Install it from https://www.python.org/downloads/
echo   (tick "Add python.exe to PATH" during install), then re-run.
echo.
pause
exit /b 1

:create
echo [2/4] Creating .venv (fresh environment) ...
if exist .venv (
    echo   .venv already exists - reusing it. (delete the folder to start fresh)
) else (
    %PYCMD% -m venv .venv
    if errorlevel 1 goto :fail
)
set "VPY=.venv\Scripts\python.exe"

echo [3/4] Upgrading pip ...
"%VPY%" -m pip install --upgrade pip -q
if errorlevel 1 goto :fail

echo [4/4] Installing pinned packages (CUDA 12.1 torch) ...
"%VPY%" -m pip install -r requirements-lock.txt
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  Verifying install ...
echo ============================================================
"%VPY%" -c "import torch, transformers, peft, albumentations, cv2, onnxruntime; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY'); print('all imports OK')"
if errorlevel 1 goto :fail

echo.
echo  Done. Activate with:    .venv\Scripts\activate
echo  Then train:             python train_all.py --data_dir dataset
echo.
pause
exit /b 0

:fail
echo.
echo  SETUP FAILED - see the error above.
pause
exit /b 1
