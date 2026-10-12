@echo off
REM ===========================================================================
REM  Start the speech engine (StyleTTS 2) - the voice behind the voice changer.
REM
REM  It loads the checkpoint onto the GPU and takes 60-120 s to be ready. This
REM  window waits until the server actually answers before saying so, then hands
REM  over to the running server. Close this window to free the GPU.
REM
REM  Everything it uses lives in this folder:
REM      engine\styletts2\.venv             the Python 3.11 that runs torch
REM      engine\styletts2\*.pth, config.yml the LibriTTS checkpoint
REM      engine\models\                     the aligner, pitch extractor, PLBERT
REM
REM  If those are missing, run Setup.bat. You do not normally need this file:
REM  RixersVoiceChanger.bat starts the engine for you from its Dashboard.
REM
REM  Pass --no-wait to skip the readiness check.
REM ===========================================================================
setlocal
cd /d "%~dp0"
title Rixers speech engine

set "PY=%~dp0.venv\Scripts\python.exe"
set "ENGINE_PY=%~dp0engine\styletts2\.venv\Scripts\python.exe"
set "WAIT=1"
if /i "%~1"=="--no-wait" set "WAIT=0"

echo.
echo   ============================================================
echo    Rixers Custom Voice Changer - speech engine
echo   ============================================================

if not exist "%ENGINE_PY%" (
    echo   [FAIL] The speech engine is not installed:
    echo          %ENGINE_PY%
    echo.
    echo   Run Setup.bat once to install it ^(about 1.6 GB, downloaded once^).
    echo.
    pause
    exit /b 1
)
if not exist "%PY%" set "PY=python"

echo    engine  : %ENGINE_PY%
echo    voice   : %~dp0voices\default.wav
echo.

REM UTF-8 is required: the model prints IPA phonemes, and the default Windows code
REM page cannot encode them - synthesis then fails with a charmap error and the
REM model silently falls back to the CPU.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM The server runs in this window's console, so Ctrl+C still stops it.
start /b "" "%ENGINE_PY%" -u "%~dp0live_transcribe\engine_tts_server.py" ^
    --reference "%~dp0voices\default.wav" %*

if "%WAIT%"=="1" (
    echo   Waiting for it to be ready ^(60-120 s is normal^)...
    "%PY%" "%~dp0launch.py" wait-tts --timeout=300
    if errorlevel 1 (
        echo.
        echo   [FAIL] The engine did not become ready. Its own output above should
        echo          say why. Try: Setup.bat check
        echo.
        pause
        exit /b 1
    )
    echo.
    echo   ============================================================
    echo    READY - the voice engine is listening on 127.0.0.1:9881
    echo    Leave this window open. Close it to free the GPU.
    echo   ============================================================
    echo.
)

REM Keep the window alive alongside the server until the user closes it.
pause
endlocal
