@echo off
setlocal EnableDelayedExpansion

echo ============================================
echo YOLO11 Pose TensorRT Build Script
echo ============================================

set BUILD_DIR=build
set CMAKE_GENERATOR="Visual Studio 17 2022"
set CMAKE_ARCH=x64

if exist %BUILD_DIR% (
    echo Cleaning existing build directory...
    rmdir /s /q %BUILD_DIR%
)

echo Creating build directory...
mkdir %BUILD_DIR%

echo.
echo Configuring CMake...
cmake -B %BUILD_DIR% -G %CMAKE_GENERATOR% -A %CMAKE_ARCH% ^
    -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" ^
    -DTensorRT_ROOT_DIR="C:/TensorRT"

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    exit /b 1
)

echo.
echo Building Release version...
cmake --build %BUILD_DIR% --config Release -j

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo ============================================
echo Build completed successfully!
echo Executables are in: %BUILD_DIR%\bin\Release
echo ============================================
