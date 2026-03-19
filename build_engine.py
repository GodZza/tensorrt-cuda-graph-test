"""
ONNX 转 TensorRT Engine 脚本 (适配固定尺寸模型)
"""

import tensorrt as trt
import argparse
import os
import sys

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, use_fp16=True):
    """
    将 ONNX 模型转换为 TensorRT Engine (固定尺寸)
    """
    print(f"Building TensorRT engine from: {onnx_path}")
    print(f"  FP16: {use_fp16}")
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    input_name = network.get_input(0).name
    input_dims = network.get_input(0).shape
    
    print(f"  Input name: {input_name}")
    print(f"  Input dims: {input_dims}")
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 mode enabled")
    else:
        print("  FP16 mode not available, using FP32")
    
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("Failed to build engine!")
        return False
    
    with open(engine_path, 'wb') as f:
        f.write(plan)
    
    print(f"Engine saved to: {engine_path}")
    print(f"  Output dims: {network.get_output(0).shape}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT Engine')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Input ONNX file path')
    parser.add_argument('--engine', type=str, default=None,
                        help='Output engine file path')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use FP16 precision')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 precision')
    
    args = parser.parse_args()
    
    engine_path = args.engine
    if engine_path is None:
        base_name = os.path.splitext(args.onnx)[0]
        engine_path = f"{base_name}.engine"
    
    success = build_engine(
        onnx_path=args.onnx,
        engine_path=engine_path,
        use_fp16=not args.fp32
    )
    
    sys.exit(0 if success else 1)
