"""
YOLO11 Pose TensorRT Python 推理脚本
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

NUM_KEYPOINTS = 17
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)
]


class YoloPoseTRT:
    def __init__(self, engine_path, conf_threshold=0.25, nms_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        input_shape = self.engine.get_tensor_shape(self.input_name)
        output_shape = self.engine.get_tensor_shape(self.output_name)
        
        self.input_size = input_shape[2]
        
        self.input_host = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
        self.output_host = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
        
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)
        
        self.stream = cuda.Stream()
        
        print(f"Engine loaded: {engine_path}")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output_shape}")
    
    def preprocess(self, image):
        self.orig_height, self.orig_width = image.shape[:2]
        
        scale = min(self.input_size / self.orig_width, self.input_size / self.orig_height)
        new_width = int(self.orig_width * scale)
        new_height = int(self.orig_height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        self.pad_x = (self.input_size - new_width) // 2
        self.pad_y = (self.input_size - new_height) // 2
        self.scale = scale
        
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[self.pad_y:self.pad_y + new_height, self.pad_x:self.pad_x + new_width] = resized
        
        input_tensor = canvas.astype(np.float32) / 255.0
        input_tensor = input_tensor[:, :, ::-1]
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.ascontiguousarray(input_tensor)
        
        return input_tensor
    
    def infer(self, image):
        input_tensor = self.preprocess(image)
        
        np.copyto(self.input_host, input_tensor.ravel())
        
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        
        self.context.set_tensor_address(self.input_name, int(self.input_device))
        self.context.set_tensor_address(self.output_name, int(self.output_device))
        self.context.execute_async_v3(self.stream.handle)
        
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        
        self.stream.synchronize()
        
        output = self.output_host.reshape(56, 8400)
        
        return self.postprocess(output)
    
    def postprocess(self, output):
        boxes = output[:4, :]
        obj_conf = output[4, :]
        keypoints = output[5:, :]
        
        mask = obj_conf > self.conf_threshold
        
        valid_boxes = boxes[:, mask]
        valid_conf = obj_conf[mask]
        valid_keypoints = keypoints[:, mask]
        
        if len(valid_conf) == 0:
            return []
        
        results = []
        for i in range(len(valid_conf)):
            cx, cy, w, h = valid_boxes[:, i]
            
            x1 = (cx - w / 2 - self.pad_x) / self.scale
            y1 = (cy - h / 2 - self.pad_y) / self.scale
            x2 = (cx + w / 2 - self.pad_x) / self.scale
            y2 = (cy + h / 2 - self.pad_y) / self.scale
            
            kpts = []
            for k in range(17):
                kx = (valid_keypoints[k * 3, i] - self.pad_x) / self.scale
                ky = (valid_keypoints[k * 3 + 1, i] - self.pad_y) / self.scale
                kc = valid_keypoints[k * 3 + 2, i]
                kpts.append((kx, ky, kc))
            
            results.append({
                'bbox': (x1, y1, x2, y2),
                'conf': valid_conf[i],
                'keypoints': kpts
            })
        
        results = self.nms(results)
        
        return results
    
    def nms(self, results):
        if len(results) == 0:
            return results
        
        results.sort(key=lambda x: x['conf'], reverse=True)
        
        keep = []
        while len(results) > 0:
            best = results.pop(0)
            keep.append(best)
            
            filtered = []
            for r in results:
                iou = self.compute_iou(best['bbox'], r['bbox'])
                if iou < self.nms_threshold:
                    filtered.append(r)
            results = filtered
        
        return keep
    
    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)


def draw_results(image, results):
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        text = f"{r['conf']:.2f}"
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for k, (kx, ky, kc) in enumerate(r['keypoints']):
            if kc > 0.5:
                kx, ky = int(kx), int(ky)
                cv2.circle(image, (kx, ky), 3, COLORS[k], -1)
        
        for i, (a, b) in enumerate(SKELETON):
            kpts = r['keypoints']
            if kpts[a][2] > 0.5 and kpts[b][2] > 0.5:
                pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                cv2.line(image, pt1, pt2, (255, 255, 255), 2)
    
    return image


def main():
    parser = argparse.ArgumentParser(description='YOLO11 Pose TensorRT Python Demo')
    parser.add_argument('--engine', type=str, default='models/yolo11n-pose.engine',
                        help='TensorRT engine path')
    parser.add_argument('--image', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.45,
                        help='NMS threshold')
    
    args = parser.parse_args()
    
    detector = YoloPoseTRT(args.engine, args.conf, args.nms)
    
    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load image: {args.image}")
        return
    
    results = detector.infer(image)
    
    output = draw_results(image.copy(), results)
    
    print(f"\nDetection results for {args.image}:")
    for i, r in enumerate(results):
        print(f"  Person {i + 1}: conf={r['conf']:.3f} bbox=[{r['bbox'][0]:.1f},{r['bbox'][1]:.1f},{r['bbox'][2]:.1f},{r['bbox'][3]:.1f}]")
    
    output_path = args.image.replace('.jpg', '_result_python.jpg')
    cv2.imwrite(output_path, output)
    print(f"\nResult saved to: {output_path}")


if __name__ == '__main__':
    main()
