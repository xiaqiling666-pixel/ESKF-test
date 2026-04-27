@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"
echo Running experiment comparison templates...
echo This path runs multiple configs and will update 03_results_experiment_* directories.
python "%ROOT%\02_src\run_experiment_batch.py"
echo.
echo Experiment batch finished. Summary is in 03_results_experiment_batch.
pause
