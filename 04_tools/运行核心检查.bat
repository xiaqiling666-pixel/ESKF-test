@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"
python -m unittest discover "%ROOT%\05_tests"
echo.
echo Core checks finished.
if /i "%~1"=="--no-pause" exit /b %errorlevel%
pause
