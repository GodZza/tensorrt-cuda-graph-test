@echo off
setlocal EnableDelayedExpansion

set PROJECT_DIR=%~dp0
set BUILD_DIR=%PROJECT_DIR%build_ninja
set BIN_DIR=%BUILD_DIR%\bin

echo ========================================
echo Building YOLO11 Pose Detection (Ninja)
echo ========================================

if exist "%BUILD_DIR%" (
    echo Removing existing build directory...
    rmdir /s /q "%BUILD_DIR%"
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

echo.
echo [1/3] Running CMake configuration with Ninja...
cmake -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89" ^
    -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" ^
    -DTensorRT_ROOT_DIR="C:/TensorRT/TensorRT-10.9.0.34" ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    ..

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Building project...
ninja

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Copying DLLs...

set CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
set TRT_BIN=C:\TensorRT\TensorRT-10.9.0.34\lib

if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

copy /Y "%CUDA_BIN%\cudart64_*.dll" "%BIN_DIR%\" 2>nul
copy /Y "%CUDA_BIN%\cublas64_*.dll" "%BIN_DIR%\" 2>nul
copy /Y "%CUDA_BIN%\cublasLt64_*.dll" "%BIN_DIR%\" 2>nul

copy /Y "%TRT_BIN%\nvinfer_10.dll" "%BIN_DIR%\" 2>nul
copy /Y "%TRT_BIN%\nvonnxparser_10.dll" "%BIN_DIR%\" 2>nul

echo.
echo ========================================
echo Build completed successfully!
echo Output: %BIN_DIR%
echo ========================================

cd /d "%PROJECT_DIR%"
endlocal
