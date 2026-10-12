@echo off
REM ===========================================================================
REM  Stop everything the launcher started.
REM
REM  Frees the GPU and releases the virtual microphone. Equivalent to closing the
REM  two windows by hand; this just does it in one go.
REM
REM  The microphone -> text stage is left alone, because it may be a browser tab
REM  you are still using. Close that window (or the tab) yourself.
REM ===========================================================================
setlocal
cd /d "%~dp0"
title voice changer - stopping

if not exist ".venv\Scripts\python.exe" (
    echo   [FAIL] Virtual environment missing.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "%~dp0launch.py" stop

echo.
echo   Done. The GPU is free and the virtual microphone is no longer being fed.
echo.
timeout /t 4 /nobreak >nul
exit /b 0
