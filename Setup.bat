@echo off
REM ===========================================================================
REM  RIXERS CUSTOM VOICE CHANGER - one-time setup
REM
REM  Double-click this after cloning. It installs everything the program needs:
REM
REM    1. this project's own environment (faster-whisper, sounddevice, numpy)
REM    2. the speech engine (StyleTTS 2) in its own Python 3.11 environment
REM    3. the model files - about 1.6 GB, downloaded once
REM    4. then loads the model and speaks a test sentence to prove it works
REM
REM  Nothing is installed globally and nothing outside this folder is touched.
REM  Safe to run again at any time: finished steps are skipped, and it will
REM  repair anything missing.
REM
REM  Options:
REM      Setup.bat --torch cpu        no usable GPU (slow, but always works)
REM      Setup.bat --torch cu118      a driver older than CUDA 12.1
REM      Setup.bat --reuse-legacy     adopt an existing Riko StyleTTS 2 install
REM      Setup.bat --skip-verify      install without the test sentence
REM      Setup.bat check              report what is installed, change nothing
REM
REM  It needs Python 3.10+ on PATH for the first step, and Python 3.11 for the
REM  speech engine. If 3.11 is missing it says so and stops before downloading
REM  anything.
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title Rixers Custom Voice Changer - setup

echo.
echo   ============================================================
echo    Rixers Custom Voice Changer - setup
echo   ============================================================
echo.

REM --- find a python to bootstrap with --------------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo   [FAIL] No Python was found on PATH.
    echo.
    echo   Install Python 3.11 from https://www.python.org/downloads/
    echo   Tick "Add python.exe to PATH" during the install, then run this again.
    echo.
    pause
    exit /b 1
)

REM --- does this project already have its own interpreter? -------------------
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PY%" set "PY="%VENV_PY%""

echo   Using: %PY%
echo.
%PY% "%~dp0setup.py" %*
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
    echo   ============================================================
    echo    Setup finished. Start the program with:
    echo.
    echo        RixersVoiceChanger.bat
    echo   ============================================================
    echo.
    timeout /t 8 /nobreak >nul
) else (
    echo   ============================================================
    echo    Setup did not finish. The messages above say what is missing.
    echo    Fix that and run Setup.bat again - it picks up where it left off.
    echo   ============================================================
    echo.
    pause
)
endlocal
exit /b %CODE%
