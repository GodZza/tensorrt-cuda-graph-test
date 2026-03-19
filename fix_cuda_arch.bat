@echo off
setlocal

echo ========================================
echo Fix CUDA 12.8 Default Architecture
echo ========================================
echo.
echo This script will modify:
echo C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.8.props
echo.
echo Change: compute_52,sm_52 -^> compute_75,sm_75
echo.

set "PROPS_FILE=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.8.props"

if not exist "%PROPS_FILE%" (
    echo ERROR: File not found: %PROPS_FILE%
    pause
    exit /b 1
)

powershell -Command "(Get-Content '%PROPS_FILE%') -replace 'compute_52,sm_52', 'compute_75,sm_75' | Set-Content '%PROPS_FILE%'"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================
    echo Successfully modified CUDA 12.8.props
    echo ========================================
) else (
    echo.
    echo ERROR: Failed to modify file. Please run as Administrator.
)

pause
