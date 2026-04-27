@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"

echo Checking Python and required packages...
python --version
if errorlevel 1 goto fail

python -c "import sys, numpy, pandas, matplotlib; print('Python executable:', sys.executable); print('numpy:', numpy.__version__); print('pandas:', pandas.__version__); print('matplotlib:', matplotlib.__version__)"
if errorlevel 1 goto fail

echo.
echo Environment check finished.
if /i "%~1"=="--no-pause" exit /b 0
pause
exit /b 0

:fail
echo.
echo Environment check failed.
echo Try: python -m pip install -r requirements.txt
if /i "%~1"=="--no-pause" exit /b 1
pause
exit /b 1
