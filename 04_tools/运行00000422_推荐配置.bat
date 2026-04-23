@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"
python "%ROOT%\02_src\main.py" "%ROOT%\01_data\config_00000422_decoded.json"
echo.
echo Run finished. Results are in 03_results_00000422_decoded_recommended.
pause
