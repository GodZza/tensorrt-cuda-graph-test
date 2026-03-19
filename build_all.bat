@echo off
setlocal

echo ============================================
echo YOLO11 Pose - Complete Build Pipeline
echo ============================================

echo.
echo [Step 1/3] Exporting YOLO11n-Pose to ONNX...
python export_onnx.py --size n --output models

if %ERRORLEVEL% neq 0 (
    echo ONNX export failed!
    exit /b 1
)

echo.
echo [Step 2/3] Building TensorRT Engine...
python build_engine.py --onnx models/yolo11n-pose.onnx --batch 16 --fp16

if %ERRORLEVEL% neq 0 (
    echo Engine build failed!
    exit /b 1
)

echo.
echo [Step 3/3] Building C++ executables...
call build_nmake.bat

if %ERRORLEVEL% neq 0 (
    echo C++ build failed!
    exit /b 1
)

echo.
echo ============================================
echo All steps completed successfully!
echo ============================================
echo.
echo To run the demo:
echo   CUDA Graph:   build_nmake\bin\yolo_cuda_graph.exe models\yolo11n-pose.engine test-img\bus.jpg
echo   Stream Async: build_nmake\bin\yolo_stream_async.exe models\yolo11n-pose.engine test-img\bus.jpg
echo.
