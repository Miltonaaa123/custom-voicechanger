@echo off
REM ===========================================================================
REM  live-transcribe launcher
REM
REM  Double-click this file, or run it from a terminal. It uses the project's
REM  own virtual environment and writes to out\transcript.txt.
REM
REM  Pass arguments straight through, for example:
REM      run.bat --list-devices
REM      run.bat --input-device 3
REM      run.bat --model tiny.en
REM ===========================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   The virtual environment is missing.
    echo   Create it once with:
    echo.
    echo       python -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

title live-transcribe
".venv\Scripts\python.exe" transcribe.py %*

echo.
echo   Transcript saved to: %~dp0out\transcript.txt
echo.
pause
endlocal
