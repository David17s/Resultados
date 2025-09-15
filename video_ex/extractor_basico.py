#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import timeit
# Assuming these imports are correctly set up in your environment
from video_extractor.unet import UNet
from video_extractor.otros.deximodel import DexiNed


# === Orientation detector settings ===
THETA_RESOLUTION = 6
KERNEL_SIZE = 9

# Model selection (0 for DexiNed, 1 for UNet)
USING_UNET = False

# Paths to your model checkpoints
DEXINED_PATH = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/dexi.pth'
UNET_PATH = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/unet.pth'


def init_orientation_detector(theta_resolution, kernel_size, device):
    """Initialize orientation detector with line kernels"""
    od = torch.nn.Conv2d(
        in_channels=1,
        out_channels=theta_resolution,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        bias=False
    ).to(device)

    for i in range(theta_resolution):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        angle = i * 180 / theta_resolution
        # Create a line in the kernel at this angle
        length = kernel_size // 2
        dx = int(np.cos(np.deg2rad(angle)) * length)
        dy = int(np.sin(np.deg2rad(angle)) * length)
        cv2.line(
            img=kernel,
            pt1=(kernel_size // 2 - dx, kernel_size // 2 - dy),
            pt2=(kernel_size // 2 + dx, kernel_size // 2 + dy),
            color=1,
            thickness=1
        )
        od.weight.data[i, 0] = torch.from_numpy(kernel)
    return od


class ShapeDetectionNode(Node):
    def __init__(self):
        super().__init__('shape_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()

        # Setup device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load model
        if not USING_UNET:
            self.model = DexiNed().to(self.device)
            self.model.load_state_dict(torch.load(DEXINED_PATH, map_location=self.device))
            self.get_logger().info('Loaded DexiNed model')
        else:
            self.model = UNet(1, 1).to(self.device)
            self.model.load_state_dict(torch.load(UNET_PATH, map_location=self.device))
            self.get_logger().info('Loaded UNet model')

        # Prepare orientation detector
        self.orientation_detector = init_orientation_detector(
            THETA_RESOLUTION,
            KERNEL_SIZE,
            self.device
        )

        self.get_logger().info('Shape detection node ready. Listening on /rgb')

    def listener_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            start = timeit.default_timer()
            output = self.detect_shapes(frame)
            elapsed = timeit.default_timer() - start

            self.get_logger().info(f'Processed frame in {elapsed:.3f}s, shapes: {output["shape_count"]}')

            cv2.imshow('Original', output['original'])
            cv2.imshow('Edges', output['edges'])
            cv2.imshow('Lines and Corners (Blue/Points)', output['lines_and_corners'])
            cv2.imshow('Detected Shapes (White)', output['final_shapes'])
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

    def detect_shapes(self, image: np.ndarray) -> dict:
        # Resize to multiple of 16
        h, w = image.shape[:2]
        res = 16
        new_w = (w // res) * res
        new_h = (h // res) * res
        img = cv2.resize(image, (new_w, new_h))

        # Convert to model input
        inp = img if not USING_UNET else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inp = torch.from_numpy(inp).to(self.device).float() / 255.0
        if USING_UNET:
            inp = inp.unsqueeze(0).unsqueeze(0)
        else:
            inp = inp.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(inp)
            orientation = self.orientation_detector(pred)
            orientation = torch.nn.functional.normalize(
                orientation - orientation.mean(1, keepdim=True),
                p=2.0,
                dim=1
            )

        edge_map = pred.detach().cpu().numpy()[0, 0]
        thresh = 0.5 if not USING_UNET else -2.5
        edges = (edge_map > thresh).astype(np.uint8) * 255

        # Hough Line detection for the 'Lines and Corners' visualization
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=15,
            maxLineGap=5
        )

        # Prepare 'lines_and_corners' image (black background, blue lines, flat points)
        lines_and_corners_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                cv2.line(lines_and_corners_img, (x1, y1), (x2, y2), (255, 0, 0), 1) # Blue color

        # Shape detection (Rectangles/Squares)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_shapes_img = img.copy() # For the final output with white lines
        shape_count = 0

        for contour in contours:
            # Approximate the contour with a polygon
            # Epsilon is critical: controls the approximation accuracy
            epsilon = 0.04 * cv2.arcLength(contour, True) 
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the approximated contour has 4 vertices (quadrilateral)
            # and is reasonably large (to filter out noise)
            if len(approx) == 4 and cv2.contourArea(approx) > 100: # Min area filter
                # You can add more checks here for "squareness" or "rectangularity"
                # For example, checking angles between segments, aspect ratio of bounding box
                # For now, we'll consider any 4-sided polygon as a detected shape.

                # Draw the detected shape on the final_shapes_img
                cv2.drawContours(final_shapes_img, [approx], 0, (255, 255, 255), 2) # White color
                shape_count += 1

                # Draw corners on the lines_and_corners_img
                for point in approx:
                    x, y = point[0]
                    cv2.circle(lines_and_corners_img, (x, y), 3, (255, 255, 255), -1) # Yellow points for corners


        return {
            'original': img,
            'edges': edges,
            'lines_and_corners': lines_and_corners_img,
            'final_shapes': final_shapes_img,
            'shape_count': shape_count
        }

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetectionNode() # Renamed node instance
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()