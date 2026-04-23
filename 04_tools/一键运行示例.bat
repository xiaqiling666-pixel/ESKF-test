@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"
python "%ROOT%\02_src\main.py"
echo.
echo Run finished. Results are in 03_results.
pause
