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
    
    REM Check for environment conflicts before activating
    if defined CONDA_PREFIX if not defined VIRTUAL_ENV (
        echo.
        echo ============================================================
        echo WARNING: Detected CONDA_PREFIX while activating venv
        echo ============================================================
        echo.
        echo A Conda environment is currently active.
        echo This can cause conflicts when building Rust extensions with maturin.
        echo.
        echo Deactivating Conda environment to avoid conflicts...
        call conda deactivate
        echo Conda environment deactivated.
        echo.
    )
    
    call venv\Scripts\activate.bat
    goto check_extensions
)

REM Check for .venv as alternative
if exist ".venv\Scripts\python.exe" (
    echo Virtual environment found. Activating...
    
    REM Check for environment conflicts before activating
    if defined CONDA_PREFIX if not defined VIRTUAL_ENV (
        echo.
        echo ============================================================
        echo WARNING: Detected CONDA_PREFIX while activating venv
        echo ============================================================
        echo.
        echo A Conda environment is currently active.
        echo This can cause conflicts when building Rust extensions with maturin.
        echo.
        echo Deactivating Conda environment to avoid conflicts...
        call conda deactivate
        echo Conda environment deactivated.
        echo.
    )
    
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

REM ============================================================
REM Check Python Architecture (64-bit required)
REM ============================================================
echo Checking Python architecture...
%PYTHON_CMD% -c "import platform; print(platform.architecture()[0])" | findstr "64bit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Detected 32-bit Python installation.
    echo ============================================================
    echo.
    echo This application requires a 64-bit Python installation.
    echo 32-bit Python can cause build failures, especially for Cython
    echo extensions that require matching architecture with Visual C++
    echo Build Tools.
    echo.
    echo Please install 64-bit Python from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to select the "x86-64" or "64-bit" installer.
    echo ============================================================
    echo.
    pause
    exit /b 1
)
echo Python architecture: 64-bit (OK)
echo.

REM ============================================================
REM Detect and Set Visual C++ Compiler Path (64-bit)
REM ============================================================
echo Detecting Visual C++ Build Tools...

REM Try to find vcvarsall.bat for various Visual Studio versions
set "VCVARSALL_FOUND="
set "VS_INSTALL_DIR="

REM Check Visual Studio 2022
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files\Microsoft Visual Studio\2022\Community"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files\Microsoft Visual Studio\2022\Professional"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)

REM Check Visual Studio 2019
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)

REM Check Visual Studio 2017
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_INSTALL_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise"
    set "VCVARSALL_FOUND=1"
    goto vcvarsall_detected
)

:vcvarsall_detected
if defined VCVARSALL_FOUND (
    echo Found Visual Studio Build Tools at:
    echo   %VS_INSTALL_DIR%
    echo.
    echo Setting up 64-bit compiler environment...
    
    REM Call vcvarsall.bat to set up the compiler environment for x64
    REM This will configure PATH and other environment variables correctly
    call "%VS_INSTALL_DIR%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
    
    if not errorlevel 1 (
        echo Visual C++ 64-bit compiler environment configured successfully.
        echo.
    ) else (
        echo Warning: Failed to configure Visual C++ environment.
        echo This may cause Cython extension build failures.
        echo.
    )
) else (
    echo Visual Studio Build Tools not found at common installation paths.
    echo.
    echo Note: Cython extension builds require Visual C++ Build Tools.
    echo If you encounter build errors later, install from:
    echo   https://visualstudio.microsoft.com/downloads/
    echo.
    echo Continuing with current environment...
    echo.
)

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

REM ============================================================
REM Check for Environment Conflicts (VIRTUAL_ENV vs CONDA_PREFIX)
REM ============================================================
if defined CONDA_PREFIX if defined VIRTUAL_ENV (
    echo.
    echo ============================================================
    echo WARNING: Detected both VIRTUAL_ENV and CONDA_PREFIX
    echo ============================================================
    echo.
    echo Both a virtual environment and Conda environment are active.
    echo This can cause conflicts when building Rust extensions with maturin,
    echo which requires only one environment to be active.
    echo.
    echo Deactivating Conda environment to avoid conflicts...
    call conda deactivate
    echo Conda environment deactivated.
    echo.
)

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
python -c "import os; import sys; pyd_files = []; exec('try: pyd_files = [f for f in os.listdir(\"anylabeling/extensions\") if f.endswith(\".pyd\")]\nexcept: pass'); sys.exit(int(not pyd_files))" 2>nul
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
            goto skip_cython_build
        )
    )
    
    REM Only attempt build if dependencies were installed successfully
    python -c "import cython; import numpy" 2>nul
    if not errorlevel 1 (
        REM ============================================================
        REM Verify Compiler Architecture Before Building
        REM ============================================================
        echo Verifying compiler architecture compatibility...
        
        REM Check if cl.exe is available in PATH
        where cl.exe >nul 2>&1
        if not errorlevel 1 (
            REM Capture cl.exe output to check for x64 architecture
            echo Checking cl.exe architecture...
            cl.exe 2>&1 | findstr /i "x64" >nul 2>&1
            if errorlevel 1 (
                REM Check for x86 architecture (32-bit) - this is a problem
                cl.exe 2>&1 | findstr /i "x86" >nul 2>&1
                if not errorlevel 1 (
                    echo.
                    echo ============================================================
                    echo ERROR: Compiler architecture mismatch detected
                    echo ============================================================
                    echo.
                    echo The cl.exe compiler in PATH is 32-bit (x86), but Python is 64-bit.
                    echo This will cause linker errors (LNK1120) during Cython builds.
                    echo.
                    for /f "tokens=*" %%i in ('where cl.exe') do (
                        echo Current cl.exe location: %%i
                    )
                    echo.
                    echo To fix this issue:
                    echo   1. Install Visual C++ Build Tools (64-bit) from:
                    echo      https://visualstudio.microsoft.com/downloads/
                    echo   2. Make sure the 64-bit compiler path comes first in PATH
                    echo   3. The expected path pattern should contain: HostX64\x64
                    echo.
                    echo Skipping Cython build to avoid errors.
                    echo The application will still work with Python fallback implementations.
                    echo ============================================================
                    echo.
                    goto skip_cython_build
                )
            ) else (
                echo Compiler architecture: 64-bit (x64) - Compatible!
                for /f "tokens=*" %%i in ('where cl.exe') do (
                    echo Using cl.exe from: %%i
                )
                echo.
            )
        ) else (
            echo Warning: cl.exe not found in PATH.
            echo Attempting to build anyway - may fail on Windows without Visual C++ Build Tools.
            echo.
        )
        
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

:skip_cython_build
echo.

:check_rust
REM Check and build Rust extensions
echo [Rust Extensions]

REM Check if Rust extensions already available
python -c "from anylabeling.rust_extensions import RUST_AVAILABLE; import sys; sys.exit(int(not RUST_AVAILABLE))" 2>nul
if not errorlevel 1 goto rust_already_available

echo Rust extensions not found. Checking for Rust toolchain...

REM Check if rustc is available
rustc --version >nul 2>&1
if errorlevel 1 goto rust_not_installed

echo Rust toolchain found. Attempting to build...

REM Install maturin if needed
python -c "import maturin" 2>nul
if not errorlevel 1 goto maturin_ready
echo Installing maturin...
pip install --quiet maturin
if errorlevel 1 goto maturin_failed

:maturin_ready
if not exist "anylabeling\rust_extensions" goto rust_dir_missing

REM Check for environment conflicts before building Rust extensions
if defined CONDA_PREFIX if defined VIRTUAL_ENV (
    echo.
    echo ============================================================
    echo WARNING: Environment conflict detected before Rust build
    echo ============================================================
    echo.
    echo Both VIRTUAL_ENV and CONDA_PREFIX are set.
    echo This can cause maturin build failures.
    echo.
    echo Attempting to deactivate Conda environment...
    call conda deactivate
    echo.
)

REM Validate required Rust files exist
if not exist "anylabeling\rust_extensions\Cargo.toml" goto rust_cargo_missing
if not exist "anylabeling\rust_extensions\src\lib.rs" goto rust_lib_missing

REM Check for Windows linker (helps diagnose link errors)
set LINKER_FOUND=0
where link.exe >nul 2>&1
if not errorlevel 1 set LINKER_FOUND=1
where lld.exe >nul 2>&1
if not errorlevel 1 set LINKER_FOUND=1

if %LINKER_FOUND%==0 (
    echo Warning: No linker detected in PATH ^(link.exe or lld.exe^).
    echo This may cause build failures on Windows.
    echo.
    echo To fix:
    echo   1. Install "Desktop development with C++" workload from Visual Studio
    echo   2. Or install "Build Tools for Visual Studio" from:
    echo      https://visualstudio.microsoft.com/downloads/
    echo.
)

echo Building Rust extensions (this may take a few minutes)...
pushd anylabeling\rust_extensions

REM Capture build output to temporary file for better error reporting
set TEMP_LOG=%TEMP%\rust_build_%RANDOM%.log
maturin develop --release > "%TEMP_LOG%" 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Rust extension build failed
    echo ============================================================
    echo.
    echo Build output:
    type "%TEMP_LOG%"
    echo.
    echo ============================================================
    echo Troubleshooting:
    echo   1. Ensure you have Visual C++ Build Tools installed
    echo   2. Try running: rustup update
    echo   3. Check if Cargo.lock has conflicts
    echo   4. Try: cargo clean (in rust_extensions directory)
    echo   5. For detailed logs, run manually:
    echo      cd anylabeling\rust_extensions
    echo      maturin develop --release
    echo.
    echo The application will still work with Python fallback implementations.
    echo ============================================================
) else (
    echo Success: Rust extensions built successfully!
)
REM Clean up temporary log file
del "%TEMP_LOG%" >nul 2>&1
popd
goto rust_done

:rust_already_available
echo Rust extensions already available. Skipping build.
goto rust_done

:rust_not_installed
echo Rust toolchain not found. Skipping Rust extensions.
echo.
echo To enable Rust extensions:
echo   1. Install Rust from https://rustup.rs/
echo   2. Restart your terminal
echo   3. Re-run this script
echo.
echo The application will still work with Python fallback implementations.
goto rust_done

:maturin_failed
echo Warning: Failed to install maturin. Skipping Rust extensions.
goto rust_done

:rust_dir_missing
echo Warning: Rust extensions directory not found. Skipping build.
goto rust_done

:rust_cargo_missing
echo Warning: Cargo.toml not found in anylabeling\rust_extensions\.
echo Cannot build Rust extensions without Cargo.toml.
echo.
echo Expected location: anylabeling\rust_extensions\Cargo.toml
goto rust_done

:rust_lib_missing
echo Warning: lib.rs not found in anylabeling\rust_extensions\src\.
echo Cannot build Rust extensions without lib.rs.
echo.
echo Expected location: anylabeling\rust_extensions\src\lib.rs
goto rust_done

:rust_done
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
