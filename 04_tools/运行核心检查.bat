@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"
echo Running clean development checks...
echo This path only runs unit tests and should not generate new result files.
python -m unittest discover "%ROOT%\05_tests"
echo.
echo Core checks finished.
if /i "%~1"=="--no-pause" exit /b %errorlevel%
pause
