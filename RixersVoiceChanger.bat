@echo off
REM ===========================================================================
REM  RIXERS CUSTOM VOICE CHANGER - the control panel
REM
REM  Opens the settings window: voices, sentence timing, your own rhythm, the
REM  chunking the engine uses, audio devices, effects, and the microphone -> text
REM  stage. Everything a running voice changer can honour is applied on the next
REM  sentence; the rest is applied the next time it starts.
REM
REM  The window has its own Start / Stop buttons, so this launcher has one job:
REM  open it. Nothing is started in the background until you press Start.
REM
REM  Options, for example:
REM      RixersVoiceChanger.bat --settings out\other_settings.json
REM ===========================================================================
setlocal
cd /d "%~dp0"
title Rixers Custom Voice Changer

set "PYW=%~dp0.venv\Scripts\pythonw.exe"
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo   [FAIL] The project virtual environment is missing:
    echo          %PY%
    echo.
    echo   Create it once with:
    echo       python -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM pythonw keeps a console window from appearing behind the control panel. It
REM has no stdout, so a startup failure is reported with a message box instead.
if exist "%PYW%" (
    start "" "%PYW%" "%~dp0rixers_gui.py" %*
    exit /b 0
)

"%PY%" "%~dp0rixers_gui.py" %*
if errorlevel 1 (
    echo.
    echo   The control panel exited with an error. The text above says why.
    pause
)
endlocal
