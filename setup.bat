@echo off
REM Wrapper for setup-environment.ps1 to enable double-click execution.
REM ExecutionPolicy Bypass applies only to this powershell.exe invocation;
REM the system-wide policy is not modified.

cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-environment.ps1" %*
set EXITCODE=%ERRORLEVEL%

echo.
echo ========================================
if %EXITCODE% equ 0 (
    echo Setup script finished successfully.
) else (
    echo Setup script exited with code %EXITCODE%.
)
echo ========================================
echo.
echo Press any key to close this window...
pause >nul

exit /b %EXITCODE%
