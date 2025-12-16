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

REM Try to find a compatible Python version (3.12, 3.11, or 3.10)
echo Searching for compatible Python version (3.10, 3.11, or 3.12)...
echo.

REM Try Python 3.12 using py launcher
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12
    goto found_python
)

REM Try Python 3.11 using py launcher
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.11
    goto found_python
)

REM Try Python 3.10 using py launcher
py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.10
    goto found_python
)

REM Check if default 'python' command is available and get its version
python --version >nul 2>&1
if errorlevel 1 (
    goto no_python_found
)

REM Get Python version from 'python' command
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i

REM Extract major.minor version (e.g., "3.12" from "3.12.0")
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

REM Check if it's Python 3.10, 3.11, or 3.12
if "%PYTHON_MAJOR%"=="3" (
    if "%PYTHON_MINOR%"=="10" (
        set PYTHON_CMD=python
        goto found_python
    )
    if "%PYTHON_MINOR%"=="11" (
        set PYTHON_CMD=python
        goto found_python
    )
    if "%PYTHON_MINOR%"=="12" (
        set PYTHON_CMD=python
        goto found_python
    )
)

:no_python_found
echo ERROR: Python 3.10, 3.11, or 3.12 is required but not found.
echo.
echo This application requires Python 3.10, 3.11, or 3.12 to work properly.
echo Please install one of these versions from https://www.python.org/downloads/
echo.
echo Note: Newer versions like Python 3.13+ are not yet supported due to
echo       dependency compatibility issues.
echo.
pause
exit /b 1

:found_python
echo Found compatible Python version:
%PYTHON_CMD% --version
echo.

REM Create virtual environment
echo Creating virtual environment...
%PYTHON_CMD% -m venv venv

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
