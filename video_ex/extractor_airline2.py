#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn as nn
import timeit
from video_extractor.unet import UNet
from video_extractor.otros.deximodel import DexiNed
from video_extractor.otros import CRG311


# === Configuration ===
THETA_RESOLUTION = 6
KERNEL_SIZE = 9
USING_UNET = False  # Set to True to use UNet instead of DexiNed

# Model paths
DEXINED_PATH = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/dexi.pth'
UNET_PATH = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/unet.pth'


def init_orientation_detector(theta_resolution, kernel_size, device):
    """Initialize orientation detector with line kernels at different angles"""
    od = nn.Conv2d(1, theta_resolution, kernel_size, 1, kernel_size//2, bias=False).to(device)
    
    for i in range(theta_resolution):
        kernel = np.zeros((kernel_size, kernel_size))
        angle = i * 180 / theta_resolution
        x = (np.cos(angle/180 * 3.1415926) * 50).astype(np.int32)
        y = (np.sin(angle/180 * 3.1415926) * 50).astype(np.int32)
        
        cv2.line(kernel, (kernel_size//2-x, kernel_size//2-y), 
                (kernel_size//2+x, kernel_size//2+y), 1, 1)
        od.weight.data[i] = torch.tensor(kernel)
    
    return od


class IntegratedLineDetectionNode(Node):
    def __init__(self):
        super().__init__('integrated_line_detection_node')
        
        # ROS2 setup
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        
        # Device setup
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load the appropriate model
        if not USING_UNET:
            self.model = DexiNed().to(self.device)
            self.model.load_state_dict(torch.load(DEXINED_PATH, map_location=self.device))
            self.get_logger().info('Loaded DexiNed model')
        else:
            self.model = UNet(1, 1).to(self.device)
            self.model.load_state_dict(torch.load(UNET_PATH, map_location=self.device))
            self.get_logger().info('Loaded UNet model')
        
        # Initialize orientation detector
        self.orientation_detector = init_orientation_detector(
            THETA_RESOLUTION, KERNEL_SIZE, self.device
        )
        
        # Buffers for C++ CRG processing
        self.temp_mem = np.zeros((50000, 2), dtype=np.int32)
        self.temp_mem2 = np.zeros((2, 300000, 2), dtype=np.int32)
        
        self.get_logger().info('Integrated line detection node initialized')

    def image_callback(self, msg: Image):
        try:
            # Convert ROS image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Process the frame
            start_time = timeit.default_timer()
            results = self.process_frame(frame)
            processing_time = timeit.default_timer() - start_time
            
            self.get_logger().info(f'Processing time: {processing_time:.3f}s, Lines detected: {results["line_count"]}')
            
            # Display all three outputs
            cv2.imshow('1. Segmentation', results['segmentation'])
            cv2.imshow('2. Edge Detection', results['edges'])
            cv2.imshow('3. Line Detection', results['lines'])
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

    def process_frame(self, image):
        """Process frame to generate the three outputs: segmentation, edges, and lines"""
        
        # Resize to multiple of 16 for model compatibility
        res = 16
        h, w = image.shape[:2]
        new_h = (h // res) * res
        new_w = (w // res) * res
        rx1 = cv2.resize(image, (new_w, new_h))
        
        # Ensure RGB format
        if len(rx1.shape) == 2:
            rx1 = cv2.cvtColor(rx1, cv2.COLOR_GRAY2RGB)
        elif rx1.shape[2] == 4:
            rx1 = cv2.cvtColor(rx1, cv2.COLOR_RGBA2RGB)
        elif rx1.shape[2] == 3:
            rx1 = cv2.cvtColor(rx1, cv2.COLOR_BGR2RGB)
        
        rx1 = np.ascontiguousarray(rx1)
        
        # Prepare input for model
        x1 = rx1.copy()
        if USING_UNET:
            x1 = cv2.cvtColor(x1, cv2.COLOR_RGB2GRAY)
        
        # Convert to tensor
        x1_tensor = torch.tensor(x1).to(self.device).float() / 255.0
        
        if USING_UNET:
            x1_tensor = x1_tensor.unsqueeze(0).unsqueeze(0)
        else:
            x1_tensor = x1_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Model inference
        with torch.no_grad():
            line_detection = self.model(x1_tensor)
            theta_des = self.orientation_detector(line_detection)
            theta_des = torch.nn.functional.normalize(
                theta_des - theta_des.mean(1, keepdim=True), p=2.0, dim=1
            )
        
        # Convert to numpy for processing
        edge_np = line_detection.detach().cpu().numpy()[0, 0]
        
        # Threshold for edge detection
        if USING_UNET:
            edge_binary = (edge_np > -2.5).astype(np.uint8) * 255
        else:
            edge_binary = (edge_np > 0.5).astype(np.uint8) * 255
        
        # Initialize output maps
        segmentation_map = np.zeros_like(edge_np, dtype=np.uint8)
        segmentation_map = np.expand_dims(segmentation_map, 2).repeat(3, 2)
        
        # Line growing using CRG
        out = np.zeros((30000, 2, 2), dtype=np.float32)
        temp_mem3 = np.zeros((30000, 2, 2), dtype=np.float32)
        
        line_count = crg.desGrow(
            segmentation_map,
            edge_binary,
            theta_des[0].detach().cpu().numpy(),
            out,
            0.9,  # demo threshold
            10,   # demo max length
            self.temp_mem,
            self.temp_mem2,
            temp_mem3,
            THETA_RESOLUTION
        )
        
        # Create line visualization
        line_image = rx1.copy()
        out = out.astype(np.int32)
        
        # Draw detected lines
        line_color = (255, 255, 0)  # Yellow lines
        for i in range(line_count):
            length = np.sqrt((out[i, 0, 0] - out[i, 1, 0])**2 + 
                           (out[i, 0, 1] - out[i, 1, 1])**2)
            if length > 15:
                cv2.line(line_image, 
                        (out[i, 0, 1], out[i, 0, 0]), 
                        (out[i, 1, 1], out[i, 1, 0]), 
                        line_color, 2)
        
        # Convert back to BGR for display
        line_image = cv2.cvtColor(line_image, cv2.COLOR_RGB2BGR)
        segmentation_bgr = cv2.cvtColor(segmentation_map, cv2.COLOR_RGB2BGR)
        
        return {
            'segmentation': segmentation_bgr,
            'edges': edge_binary,
            'lines': line_image,
            'line_count': line_count
        }

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedLineDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()