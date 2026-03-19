"""
ONNX 转 TensorRT Engine 脚本 (支持动态 Batch Size 和动态输入尺寸)
"""

import tensorrt as trt
import argparse
import os
import sys

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, use_fp16=True, min_batch=1, opt_batch=8, max_batch=16, imgsz=640):
    """
    将 ONNX 模型转换为 TensorRT Engine (支持动态 Batch Size 和动态输入尺寸)
    """
    print(f"Building TensorRT engine from: {onnx_path}")
    print(f"  FP16: {use_fp16}")
    print(f"  Batch size: min={min_batch}, opt={opt_batch}, max={max_batch}")
    print(f"  Image size: {imgsz}")
    
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
    
    profile = builder.create_optimization_profile()
    
    if input_dims[2] == -1 or input_dims[3] == -1:
        print(f"  Dynamic input size detected, using fixed size: {imgsz}")
        profile.set_shape(
            input_name,
            (min_batch, input_dims[1], imgsz, imgsz),
            (opt_batch, input_dims[1], imgsz, imgsz),
            (max_batch, input_dims[1], imgsz, imgsz)
        )
    else:
        print(f"  Using fixed input dimensions: {input_dims}")
        profile.set_shape(
            input_name,
            (min_batch, input_dims[1], input_dims[2], input_dims[3]),
            (opt_batch, input_dims[1], input_dims[2], input_dims[3]),
            (max_batch, input_dims[1], input_dims[2], input_dims[3])
        )
    
    config.add_optimization_profile(profile)
    print(f"  Dynamic batch enabled for input: {min_batch}-{max_batch}")
    
    for i in range(network.num_outputs):
        output = network.get_output(i)
        output_name = output.name
        output_dims = output.shape
        print(f"  Output {i}: {output_name}, dims: {output_dims}")
    
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
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX')
    parser.add_argument('--onnx', type=str, required=True, help='ONNX model path')
    parser.add_argument('--output', type=str, required=True, help='Output engine path')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16')
    parser.add_argument('--min-batch', type=int, default=1, help='Minimum batch size')
    parser.add_argument('--opt-batch', type=int, default=4, help='Optimal batch size')
    parser.add_argument('--max-batch', type=int, default=16, help='Maximum batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    
    args = parser.parse_args()
    
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        use_fp16=args.fp16,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        imgsz=args.imgsz
    )

