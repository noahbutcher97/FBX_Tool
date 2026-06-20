@echo off
REM FBX Tool launcher - activates the .fbxenv virtual environment and starts the GUI.
REM Pause-on-error pattern: window stays open only if launch fails, so successful
REM double-clicks feel like launching any other desktop app.

cd /d "%~dp0"

if not exist ".fbxenv\Scripts\activate.bat" (
    echo ERROR: Virtual environment .fbxenv was not found.
    echo.
    echo Run setup.bat first to create the environment and install dependencies.
    echo.
    pause
    exit /b 1
)

call ".fbxenv\Scripts\activate.bat"

python -m fbx_tool %*
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% neq 0 (
    echo.
    echo ========================================
    echo FBX Tool exited with error code %EXITCODE%.
    echo ========================================
    pause
)

exit /b %EXITCODE%
