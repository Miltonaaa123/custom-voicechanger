@echo off
REM ===========================================================================
REM  START EVERYTHING
REM
REM  Brings up the whole voice changer:
REM    1. the StyleTTS 2 speech server   (its own window, ~60-90 s to load)
REM    2. waits until it is actually ready
REM    3. the voice changer              (its own window, watches the transcript)
REM    4. closes THIS window once both are up
REM
REM  The two windows that remain are the running program. Close either to stop
REM  that part; closing the speech server window frees the GPU.
REM
REM  This does NOT start the microphone -> text stage. Run one of these as well,
REM  in the window it opens:
REM      run-google.bat     browser speech recognition (accurate, needs Chrome)
REM      run.bat            local Whisper on the GPU (works offline)
REM
REM  Options:
REM      start-all.bat --no-preflight      skip the environment check
REM      start-all.bat --no-voice-changer  only start the speech server
REM      start-all.bat --with-picker       also open the voice picker window
REM      start-all.bat --voice narrator    start with a particular voice
REM      start-all.bat --mode chunk        words per synthesis call
REM ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
title voice changer launcher

set "PY=%~dp0.venv\Scripts\python.exe"
set "PREFLIGHT=1"
set "STARTVC=1"
set "PICKER=0"
set "VCOPTIONS="

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--no-preflight" set "PREFLIGHT=0"
if /i "%~1"=="--no-voice-changer" set "STARTVC=0"
if /i "%~1"=="--with-picker" set "PICKER=1"
if /i "%~1"=="--voice" set "VCOPTIONS=!VCOPTIONS! --voice %~2" & shift
if /i "%~1"=="--mode" set "VCOPTIONS=!VCOPTIONS! --mode %~2" & shift
shift
goto parse
:parsed

echo.
echo   ============================================================
echo    Voice changer - starting everything
echo   ============================================================
echo.

if not exist "%PY%" (
    echo   [FAIL] The project virtual environment is missing:
    echo          %PY%
    echo   Create it once with:  python -m venv .venv
    echo                        .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if "%PREFLIGHT%"=="1" (
    "%PY%" "%~dp0launch.py" check
    if errorlevel 1 (
        echo.
        echo   [FAIL] Fix the problems above first. Press any key to close.
        echo.
        pause
        exit /b 1
    )
    echo.
)

REM -- 1. the speech server, in its own window ------------------------------
echo   [1/3] Starting the StyleTTS 2 speech server in a new window...
start "styletts2 voice server" /d "%~dp0" cmd /c "run-tts.bat --no-wait"

REM -- 2. wait for it ---------------------------------------------------------
echo   [2/3] Waiting for the voice engine to finish loading ^(60-90 s^)...
"%PY%" "%~dp0launch.py" wait-tts --timeout=300
if errorlevel 1 (
    echo.
    echo   [FAIL] The speech server never became ready.
    echo          Look at the "styletts2 voice server" window for its error.
    echo          Press any key to close this launcher.
    echo.
    pause
    exit /b 1
)

if "%STARTVC%"=="0" (
    echo.
    echo   ============================================================
    echo    DONE - the voice engine is up. This window will now close.
    echo   ============================================================
    timeout /t 3 /nobreak >nul
    exit /b 0
)

REM -- 3. the voice changer, in its own window -------------------------------
echo   [3/3] Starting the voice changer in a new window...
start "voice-changer" /d "%~dp0" cmd /c "run-voice-changer.bat!VCOPTIONS!"

if "%PICKER%"=="1" (
    echo         and the voice picker...
    start "" /d "%~dp0" ".venv\Scripts\pythonw.exe" voice_picker.py
)

echo.
echo   ============================================================
echo    EVERYTHING IS UP
echo.
echo      "styletts2 voice server"  the voice engine (close to free the GPU)
echo      "voice-changer"           speaks your transcript in another voice
if "%PICKER%"=="1" echo      "Voice picker"            switch voices while it runs
echo.
echo    Still needed: the microphone -^> text stage. Run run-google.bat
echo    or run.bat, then speak and watch out\transcript.txt fill up.
echo.
echo    In Discord or a game, choose this microphone:
echo      CABLE Output (VB-Audio Virtual Cable)
echo.
echo    To change voice while it runs, use pick-voice.bat (no typing needed).
echo   ============================================================
echo.
echo   Closing this launcher in 8 s...
timeout /t 8 /nobreak >nul
exit /b 0
