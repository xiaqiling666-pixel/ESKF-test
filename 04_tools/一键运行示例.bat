@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"
echo Running full demo pipeline...
echo This path will update files under 03_results. For daily development, use 04_tools\运行核心检查.bat first.
python "%ROOT%\02_src\main.py"
echo.
echo Run finished. Results are in 03_results.
pause
