"""
YOLO11 Pose 模型导出脚本 - 固定尺寸版本
将 YOLO11 Pose 模型导出为 ONNX 格式，"""

from ultralytics import YOLO
import argparse
import os
import shutil

def export_yolo11_pose(model_size='n', output_dir='models'):
    """
    导出 YOLO11 Pose 模型 (固定尺寸)
    """
    model_name = f'yolo11{model_size}-pose'
    print(f"Loading {model_name}...")
    
    model = YOLO(f'{model_name}.pt')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting to ONNX with fixed input size...")
    
    model.export(
        format='onnx',
        imgsz=640,
        simplify=False,
        dynamic=False,
        opset=12,
    )
    
    src_path = f'{model_name}.onnx'
    dst_path = os.path.join(output_dir, f'{model_name}.onnx')
    
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Model exported to: {dst_path}")
    else:
        print(f"Warning: ONNX file not found at {src_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export YOLO11 Pose to ONNX')
    parser.add_argument('--size', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size: n/s/m/l/x')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory')
    
    args = parser.parse_args()
    
    export_yolo11_pose(
        model_size=args.size,
        output_dir=args.output
    )
