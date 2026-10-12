@echo off
REM ===========================================================================
REM  Voice picker - choose the cloned voice in a window, no typing needed.
REM
REM  The voice changer reads voices\voice.txt before every sentence, so picking a
REM  voice here takes effect on the next thing you say. Start the speech engine
REM  first (start-all.bat or run-tts.bat), otherwise the status line says so.
REM ===========================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo   [FAIL] Virtual environment missing - run run.bat once to create it.
    pause
    exit /b 1
)

REM pythonw so no console window appears behind the picker.
start "" ".venv\Scripts\pythonw.exe" voice_picker.py %*
endlocal
