@echo off
setlocal

echo ============================================
echo YOLO11 Pose - Performance Comparison
echo ============================================

set ENGINE_PATH=models\yolo11n-pose.engine
set IMAGE_PATH=test-img\bus.jpg
set ITERATIONS=200

if not exist %ENGINE_PATH% (
    echo Engine file not found: %ENGINE_PATH%
    echo Please run build_all.bat first to build the engine.
    exit /b 1
)

echo.
echo Running benchmarks with %ITERATIONS% iterations...
echo.

echo ============================================
echo [1/2] CUDA Stream Async Version
echo ============================================
build_nmake\bin\yolo_stream_async.exe %ENGINE_PATH% %IMAGE_PATH% %ITERATIONS%

echo.
echo ============================================
echo [2/2] CUDA Graph Version
echo ============================================
build_nmake\bin\yolo_cuda_graph.exe %ENGINE_PATH% %IMAGE_PATH% %ITERATIONS%

echo.
echo ============================================
echo Performance comparison completed!
echo ============================================
