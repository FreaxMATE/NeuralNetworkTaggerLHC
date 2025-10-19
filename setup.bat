@echo off
setlocal enabledelayedexpansion

REM Setup script for Windows (cmd). Creates a venv and installs requirements.
REM Usage: setup.bat

set PY=python
set VENV_DIR=.venv

where %PY% >nul 2>&1
if errorlevel 1 (
  echo Python not found. Please install Python 3.10+ and retry.
  exit /b 1
)

echo Creating virtual environment in %VENV_DIR% ...
%PY% -m venv %VENV_DIR%

call %VENV_DIR%\Scripts\activate

python -m pip install --upgrade pip

if exist requirements.txt (
  echo Installing Python dependencies from requirements.txt ...
  pip install -r requirements.txt
) else (
  echo requirements.txt not found; installing minimal deps ...
  pip install numpy matplotlib scikit-learn torch
)

echo.
echo Done. Activate your environment with:
echo   %VENV_DIR%\Scripts\activate

echo Then run:
echo   python data-exercise-template-v7.py
