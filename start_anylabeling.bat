@echo off
REM ============================================================
REM AnyLabeling Quick Start Script
REM Creates a virtual environment, installs dependencies, and runs the app
REM ============================================================

echo ========================================
echo    AnyLabeling Quick Start
echo ========================================
echo.

REM Get the directory where this script is located
set PROJECT_ROOT=%~dp0

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Check if venv already exists
if exist "venv\Scripts\python.exe" (
    echo Virtual environment found. Activating...
    call venv\Scripts\activate.bat
    goto run_app
)

REM Check for .venv as alternative
if exist ".venv\Scripts\python.exe" (
    echo Virtual environment found. Activating...
    call .venv\Scripts\activate.bat
    goto run_app
)

REM No venv found, create one
echo No virtual environment found. Creating one...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher and try again.
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install the package
echo.
echo Installing AnyLabeling and dependencies...
echo This may take a few minutes...
echo.
pip install -e .

if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Installation complete!
echo.

:run_app
echo ========================================
echo    Starting AnyLabeling...
echo ========================================
echo.

REM Run the application
python anylabeling\app.py %*

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
