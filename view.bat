@echo off
REM Open the always-on-top window that shows the transcript as it is written.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Virtual environment missing - run run.bat first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" viewer.py %*
endlocal
