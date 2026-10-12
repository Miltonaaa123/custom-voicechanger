@echo off
REM ===========================================================================
REM  Google speech -> text file  (free, uses Chrome/Edge's built-in recognizer)
REM
REM  Starts a tiny local server, opens the page in Chrome, and writes whatever
REM  you say to out\transcript.txt. Keep the tab open while you dictate.
REM
REM  Options, for example:
REM      run-google.bat --language en-GB
REM      run-google.bat --append
REM      run-google.bat --no-open
REM ===========================================================================
setlocal
cd /d "%~dp0"
title google-stt

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   [FAIL] The virtual environment is missing. Create it once with:
    echo.
    echo       python -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo   ============================================================
echo    Google speech -^> text file
echo   ============================================================
echo    Starts a local page server and opens it in Chrome. Keep that
echo    tab open while you dictate: it writes out\transcript.txt.
echo.

REM The server blocks, so it runs in this window; launch.py reports readiness
REM from a second process a moment later.
start /b "" "%~dp0.venv\Scripts\python.exe" "%~dp0launch.py" wait-stt --timeout=60

".venv\Scripts\python.exe" google_stt.py %*

echo.
echo   Transcript saved to: %~dp0out\transcript.txt
pause
endlocal
