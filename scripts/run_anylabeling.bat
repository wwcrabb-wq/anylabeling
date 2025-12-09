@echo off
REM Simple batch script to run AnyLabeling application on Windows
REM This script will use a virtual environment if present, or run directly

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Check for virtual environment and activate if present
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    echo Activating virtual environment...
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
    set PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe
) else if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    echo Activating virtual environment...
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
    set PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe
) else (
    set PYTHON_EXE=python
)

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Run the application
echo Starting AnyLabeling...
%PYTHON_EXE% anylabeling\app.py %*
