@echo off
REM ===========================================================================
REM  Voice changer: your speech -> another voice -> virtual microphone.
REM
REM  Start the speech engine first with run-tts.bat (or use start-all.bat, which
REM  does both). This window reports when it is actually connected, so there is a
REM  definite moment where you know it is ready.
REM
REM  In Discord / OBS / a game, choose the microphone:
REM        CABLE Output (VB-Audio Virtual Cable)
REM
REM  Options, for example:
REM      run-voice-changer.bat --monitor
REM      run-voice-changer.bat --voice narrator
REM      run-voice-changer.bat --mode chunk
REM      run-voice-changer.bat --list-voices
REM ===========================================================================
setlocal
cd /d "%~dp0"
title voice-changer

if not exist ".venv\Scripts\python.exe" (
    echo   [FAIL] Virtual environment missing - run run.bat once to create it.
    pause
    exit /b 1
)

REM --list-voices just prints the folder, so the connection check is skipped.
if /i "%~1"=="--list-voices" (
    ".venv\Scripts\python.exe" voice_changer.py --list-voices
    pause
    exit /b 0
)

echo.
echo   ============================================================
echo    Voice changer
echo   ============================================================
echo    transcript : %~dp0out\transcript.txt
echo    virtual mic: apps record from 'CABLE Output (VB-Audio Virtual Cable)'
echo.
echo   Checking the speech server...
".venv\Scripts\python.exe" "%~dp0launch.py" wait-tts --timeout=20
if errorlevel 1 (
    echo.
    echo   The voice engine is not up yet. Start it with run-tts.bat first
    echo   ^(or use start-all.bat, which starts everything in order^).
    echo.
    pause
    exit /b 1
)
echo.
echo   ============================================================
echo    READY - speak into your transcription program.
echo    Ctrl+C in this window stops it.
echo   ============================================================
echo.

".venv\Scripts\python.exe" voice_changer.py %*

echo.
echo   Voice changer stopped.
pause
endlocal
