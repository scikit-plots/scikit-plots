REM @echo on
REM REM -----------------------------------------------------------------------------
REM REM Windows build script for Conda (bld.bat)
REM REM -----------------------------------------------------------------------------
REM REM Inspired by the numpy-feedstock build script:
REM REM https://github.com/conda-forge/numpy-feedstock/blob/main/recipe/bld.bat
REM REM
REM REM Notes:
REM REM - Comment '::' must be at the start of a line (not indented), or CMD may interpret it as a label and behave strangely.
REM REM - Use 'REM' for safe comments anywhere in the script
REM REM - Use '|| exit /b N' for consistent error handling
REM REM -----------------------------------------------------------------------------
REM
REM REM Ensure submodules are initialized (e.g., if using git submodules)
REM REM git submodule update --init --recursive || exit /b 1
REM git submodule update --init --recursive || exit /b 0
REM
REM REM 🚫 You should list each of the submodules in the source section.
REM REM Example: https://github.com/conda-forge/tomopy-feedstock/blob/fc6617f1a97e866ff3d78c67c71b5d9fa76bc4fc/recipe/meta.yaml#L42
REM
REM
REM
REM REM Create a clean build directory
REM mkdir builddir || exit /b 0
REM
REM REM Build a wheel without isolation and skipping dependency checks
REM REM Flags:
REM REM   -w: build wheel
REM REM   -n: no isolation
REM REM   -x: skip dependency checks
REM REM %PYTHON% -m build -w -n -x -Cbuilddir=builddir
REM %PYTHON% -m build -w -n -x ^
REM     -Cbuilddir=builddir
REM if %ERRORLEVEL% neq 0 exit /b 1
REM
REM REM ---------------------------------------------------------------------------
REM REM Install generated wheel(s) from the dist/ folder
REM REM ---------------------------------------------------------------------------
REM
REM REM Ensure the dist folder exists
REM if not exist dist (
REM     echo ERROR: No dist/ directory found.
REM     exit /b 1
REM )
REM
REM REM Get a list of .whl files (full paths)
REM set "WHEEL_FOUND=0"
REM
REM REM Additional build commands, e.g., python setup.py install
REM REM `pip install dist\*.whl` does not work on windows,
REM REM so use a loop; there's only one wheel in dist/ anyway
REM REM for %%f in (...)	Iterate over files or hardcoded items
REM REM for /f %%f in ('command')	Iterate over command output (or lines in a file)
REM for /f "delims=" %%f in ('dir /b /s dist\*.whl 2^>nul') do (
REM     echo [INFO] Installing wheel: %%f
REM     pip install "%%f"
REM     if errorlevel 1 (
REM         echo [ERROR] pip install failed for: %%f
REM         exit /b 1
REM     )
REM     set WHEEL_FOUND=1
REM )
REM
REM REM Check if any wheel was installed
REM IF "%WHEEL_FOUND%"=="0" (
REM     echo [ERROR] No wheel files found in dist/.
REM     exit /b 0
REM )
REM
REM REM Success
REM echo [SUCCESS] Wheel(s) installed successfully.
REM
REM exit /b 0
