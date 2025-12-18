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
    goto check_extensions
)

REM Check for .venv as alternative
if exist ".venv\Scripts\python.exe" (
    echo Virtual environment found. Activating...
    call .venv\Scripts\activate.bat
    goto check_extensions
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

:check_extensions
REM ============================================================
REM Build Optional Performance Extensions
REM ============================================================
echo Checking for optional performance extensions...
echo.

REM Check and build Cython extensions
echo [Cython Extensions]
python -c "import os; import sys; pyd_files = [f for f in os.listdir('anylabeling/extensions') if f.endswith('.pyd')] if os.path.exists('anylabeling/extensions') else []; sys.exit(0 if pyd_files else 1)" 2>nul
if errorlevel 1 (
    echo Cython extensions not found. Attempting to build...
    
    REM Check if cython and numpy are installed
    python -c "import cython; import numpy" 2>nul
    if errorlevel 1 (
        echo Installing build dependencies: cython and numpy...
        pip install --quiet cython numpy
        if errorlevel 1 (
            echo Warning: Failed to install cython/numpy. Skipping Cython extensions.
            echo.
        )
    )
    
    REM Only attempt build if dependencies were installed successfully
    python -c "import cython; import numpy" 2>nul
    if not errorlevel 1 (
        REM Attempt to build Cython extensions
        echo Building Cython extensions...
        python anylabeling/extensions/setup_extensions.py build_ext --inplace 2>nul
        if errorlevel 1 (
            echo Warning: Cython extension build failed.
            echo Note: On Windows, this requires Microsoft Visual C++ Build Tools.
            echo Download from: https://visualstudio.microsoft.com/downloads/
            echo.
            echo The application will still work with Python fallback implementations.
        ) else (
            echo Success: Cython extensions built successfully!
        )
    )
) else (
    echo Cython extensions already built. Skipping build.
)
echo.

:check_rust
REM Check and build Rust extensions
echo [Rust Extensions]
REM Try to import and check RUST_AVAILABLE flag (any import error means extensions not available)
python -c "from anylabeling.rust_extensions import RUST_AVAILABLE; import sys; sys.exit(0 if RUST_AVAILABLE else 1)" 2>nul
if errorlevel 1 (
    echo Rust extensions not found. Checking for Rust toolchain...
    
    REM Check if rustc is available
    rustc --version >nul 2>&1
    if errorlevel 1 (
        echo Rust toolchain not found. Skipping Rust extensions.
        echo.
        echo To enable Rust extensions:
        echo   1. Install Rust from https://rustup.rs/
        echo   2. Restart your terminal
        echo   3. Re-run this script
        echo.
        echo The application will still work with Python fallback implementations.
    ) else (
        echo Rust toolchain found. Attempting to build...
        
        REM Check if maturin is installed
        python -c "import maturin" 2>nul
        if errorlevel 1 (
            echo Installing maturin...
            pip install --quiet maturin
            if errorlevel 1 (
                echo Warning: Failed to install maturin. Skipping Rust extensions.
                echo.
            )
        )
        
        REM Only attempt build if maturin was installed and directory exists
        python -c "import maturin" 2>nul
        if not errorlevel 1 (
            if exist "anylabeling\rust_extensions" (
                REM Build Rust extensions
                echo Building Rust extensions (this may take a few minutes)...
                cd anylabeling\rust_extensions
                maturin develop --release --quiet 2>nul
                if errorlevel 1 (
                    echo Warning: Rust extension build failed.
                    echo The application will still work with Python fallback implementations.
                ) else (
                    echo Success: Rust extensions built successfully!
                )
                cd ..\..
            ) else (
                echo Warning: Rust extensions directory not found. Skipping build.
            )
        )
    )
) else (
    echo Rust extensions already available. Skipping build.
)
echo.
echo ========================================

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
