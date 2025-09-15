#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import timeit
import math
from itertools import combinations
from video_extractor.unet import UNet
from video_extractor.otros.deximodel import DexiNed

# === Orientation detector settings ===
THETA_RESOLUTION = 6
KERNEL_SIZE = 9

# Model selection (0 for DexiNed, 1 for UNet)
USING_UNET = False

# Paths to your model checkpoints
DEXINED_PATH = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/dexi.pth'
UNET_PATH   = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/unet.pth'

# Rectangle detection parameters
ELONGATION_FACTOR = 1.5
RECTS_MIN_AREA_FACTOR = 0.001
RECTS_MAX_AREA_FACTOR = 0.1
COMPACTNESS_THRESHOLD = 0.4
LENGTH_DIFF_THRESHOLD = 0.5
NMS_THRESHOLD = 10
LINE_DRAWING_WIDTH = 0.5
RECT_DRAWING_WIDTH = 1
DRAW_GRAPH = True
DRAW_TRAPEZOIDS = False
WHITENESS_THRESHOLD = 0.7
ANGLE_THRESHOLD = 0.1
EXECUTION_PERIOD = 0.1
MAX_NUM_TRAPEZOIDS = 50

def init_orientation_detector(theta_resolution, kernel_size, device):
    """Initialize orientation detector with line kernels"""
    od = torch.nn.Conv2d(
        in_channels=1,
        out_channels=theta_resolution,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size//2,
        bias=False
    ).to(device)

    for i in range(theta_resolution):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        angle = i * 180 / theta_resolution
        # Create a line in the kernel at this angle
        length = kernel_size//2
        dx = int(np.cos(np.deg2rad(angle)) * length)
        dy = int(np.sin(np.deg2rad(angle)) * length)
        cv2.line(
            img=kernel,
            pt1=(kernel_size//2 - dx, kernel_size//2 - dy),
            pt2=(kernel_size//2 + dx, kernel_size//2 + dy),
            color=1,
            thickness=1
        )
        od.weight.data[i, 0] = torch.from_numpy(kernel)
    return od


class LineDetectionNode(Node):
    def __init__(self):
        super().__init__('line_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.last_execution = self.get_clock().now()

        # Setup device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load model
        if not USING_UNET:
            self.model = DexiNed().to(self.device)
            self.model.load_state_dict(torch.load(DEXINED_PATH, map_location=self.device))
            self.get_logger().info('Loaded DexiNed model')
        else:
            self.model = UNet(1,1).to(self.device)
            self.model.load_state_dict(torch.load(UNET_PATH, map_location=self.device))
            self.get_logger().info('Loaded UNet model')
        
        # Prepare orientation detector
        self.orientation_detector = init_orientation_detector(
            THETA_RESOLUTION,
            KERNEL_SIZE,
            self.device
        )

        # Initialize variables for rectangle detection
        self.lines = []
        self.intersections = []
        self.trapezoids = []
        self.adjacency_matrix = []
        self.image = None
        self.gray = None
        self.lines_img = None
        self.graph_img = None

        self.get_logger().info('Line and rectangle detection node ready. Listening on /rgb')

    def listener_callback(self, msg: Image):
        try:
            now = self.get_clock().now()
            if (now - self.last_execution).nanoseconds / 1e9 < EXECUTION_PERIOD:
                return
            self.last_execution = now

            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            start = timeit.default_timer()

            
            # First detect lines
            line_output = self.detect_lines(frame)
            
            # Then detect rectangles using the detected lines
            self.detect_rectangles(line_output)
            
            elapsed = timeit.default_timer() - start
            self.get_logger().info(f'Processed frame in {elapsed:.3f}s, lines: {line_output["line_count"]}')

            
            # Show results
            cv2.imshow('Original', line_output['original'])
            cv2.imshow('Edges', line_output['edges'])
            cv2.imshow('Lines', line_output['result'])
            
            if DRAW_TRAPEZOIDS and hasattr(self, 'rect_img'):
                cv2.imshow('Rectangles', self.rect_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

    def detect_lines(self, image: np.ndarray) -> dict:
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
            inp = inp.permute(2,0,1).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(inp)
            orientation = self.orientation_detector(pred)
            orientation = torch.nn.functional.normalize(
                orientation - orientation.mean(1, keepdim=True),
                p=2.0,
                dim=1
            )

        edge_map = pred.detach().cpu().numpy()[0,0]
        thresh = 0.5 if not USING_UNET else -2.5
        edges = (edge_map > thresh).astype(np.uint8) * 255

        # Hough Line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=15,
            maxLineGap=5
        )

        result = img.copy()
        count = 0
        if lines is not None:
            count = len(lines)
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                cv2.line(result, (x1, y1), (x2, y2), (0,255,255), 2)
            
            # Store lines for rectangle detection
            self.lines = [np.array([[x1, y1, x2, y2]], dtype=np.float32) for [[x1, y1, x2, y2]] in lines]

        return {
            'original': img,
            'edges': edges,
            'result': result,
            'line_count': count
        }

    def detect_rectangles(self, line_output):
        """Detect rectangles using the detected lines"""
        self.image = line_output['original']
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.lines_img = np.zeros(self.gray.shape, dtype=np.uint8)
        self.graph_img = np.zeros((self.gray.shape[0], self.gray.shape[1], 3), dtype=np.uint8)
        
        # Paint the lines on the black image
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(self.lines_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                    255, LINE_DRAWING_WIDTH, cv2.LINE_AA)
        
        # Gaussian blur
        self.lines_img = cv2.GaussianBlur(self.lines_img, (5, 5), 0)
        
        # Process for rectangle detection
        self.elongate_lines()
        self.find_intersections()
        self.create_graph()
        self.extract_trapezoids()
        
        if DRAW_TRAPEZOIDS:
            self.rect_img = self.image.copy()
            self.draw_trapezoids_func()

    def elongate_lines(self):
        """Function to increase the length of the lines"""
        for i, line in enumerate(self.lines):
            x1, y1, x2, y2 = line[0]
            
            dx = x2 - x1
            dy = y2 - y1
            
            extension = (ELONGATION_FACTOR - 1) / 2
            
            self.lines[i] = np.array([[
                x1 - dx * extension,
                y1 - dy * extension,
                x2 + dx * extension,
                y2 + dy * extension
            ]], dtype=np.float32)

    def is_white_line(self, p1, p2):
        """Function to check if a line is sufficiently white"""
        line_points = self.get_line_points(p1, p2)
        white_count = 0
        
        for point in line_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < self.lines_img.shape[1] and 0 <= y < self.lines_img.shape[0]:
                if self.lines_img[y, x] > 64:
                    white_count += 1
        
        return white_count >= len(line_points) * WHITENESS_THRESHOLD

    def get_line_points(self, p1, p2):
        """Get points along a line"""
        x1, y1 = p1
        x2, y2 = p2
        
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy)
        
        if steps == 0:
            return [p1]
        
        x_inc = (x2 - x1) / steps
        y_inc = (y2 - y1) / steps
        
        for i in range(int(steps) + 1):
            x = x1 + i * x_inc
            y = y1 + i * y_inc
            points.append([x, y])
        
        return points

    def intersect(self, line1, line2):
        """Function to check if two line segments intersect"""
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]
        
        p = np.array([x1, y1])
        q = np.array([x3, y3])
        r = np.array([x2 - x1, y2 - y1])
        s = np.array([x4 - x3, y4 - y3])
        
        rxs = np.cross(r, s)
        qp = q - p
        
        if abs(rxs) < 1e-10:  # Lines are parallel
            return False, None
        
        t = np.cross(qp, s) / rxs
        u = np.cross(qp, r) / rxs
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p + t * r
            return True, (int(round(intersection[0])), int(round(intersection[1])))
        
        return False, None

    def trapezoid_line_test(self, a, b, c, d):
        """Function to check if the difference between the shortest line and the longest is less than a threshold"""
        ab = np.linalg.norm(np.array(a) - np.array(b))
        bc = np.linalg.norm(np.array(b) - np.array(c))
        cd = np.linalg.norm(np.array(c) - np.array(d))
        da = np.linalg.norm(np.array(d) - np.array(a))
        
        min_length = min(ab, bc, cd, da)
        max_length = max(ab, bc, cd, da)
        
        if min_length / max_length < LENGTH_DIFF_THRESHOLD:
            return False
        
        return True

    def trapezoid_area_test(self, a, b, c, d, min_area, max_area):
        """Function to calculate the area of a trapezoid using shoelace formula"""
        a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)
        
        trapezoid_area = abs((a[0] * b[1] - a[1] * b[0]) + 
                           (b[0] * c[1] - b[1] * c[0]) + 
                           (c[0] * d[1] - c[1] * d[0]) + 
                           (d[0] * a[1] - d[1] * a[0])) / 2.0
        
        if trapezoid_area < min_area or trapezoid_area > max_area:
            return False
        
        perimeter = (np.linalg.norm(a - b) + np.linalg.norm(b - c) + 
                    np.linalg.norm(c - d) + np.linalg.norm(d - a))
        
        pp = 4.0 * math.pi * trapezoid_area / (perimeter * perimeter)
        if pp < COMPACTNESS_THRESHOLD:
            return False
        
        return True

    def trapezoid_angles_test(self, a, b, c, d):
        """Function to check if the angles of a trapezoid are sufficiently close to 90 degrees"""
        return True

    def non_maximum_suppression_points(self):
        """Function to remove close intersections"""
        filtered_intersections = []
        suppressed = [False] * len(self.intersections)
        
        for i in range(len(self.intersections)):
            if suppressed[i]:
                continue
            
            close_intersections = []
            for j in range(i + 1, len(self.intersections)):
                dist = np.linalg.norm(np.array(self.intersections[i]) - np.array(self.intersections[j]))
                if dist < NMS_THRESHOLD:
                    suppressed[j] = True
                    close_intersections.append(self.intersections[j])
            
            if close_intersections:
                all_points = [self.intersections[i]] + close_intersections
                avg_x = sum(p[0] for p in all_points) / len(all_points)
                avg_y = sum(p[1] for p in all_points) / len(all_points)
                filtered_intersections.append((int(avg_x), int(avg_y)))
            else:
                filtered_intersections.append(self.intersections[i])
        
        self.intersections = filtered_intersections

    def reduce_saturation_brightness(self, saturation_scale, value_scale):
        """Function to reduce the saturation and brightness of an image"""
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation_scale
        hsv_image[:, :, 2] = hsv_image[:, :, 2] * value_scale
        self.image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def draw_graph_func(self):
        """Function to draw the graph"""
        for i in range(len(self.adjacency_matrix)):
            for j in range(i + 1, len(self.adjacency_matrix)):
                if self.adjacency_matrix[i][j]:
                    cv2.line(self.graph_img, self.intersections[i], self.intersections[j], 
                            (0, 255, 0), 1, cv2.LINE_AA)
        
        for intersection in self.intersections:
            cv2.circle(self.graph_img, intersection, 2, (255, 255, 255), -1)

    def draw_trapezoids_func(self):
        """Function to draw the trapezoids"""
        color = (255, 255, 255)
        for trapezoid in self.trapezoids:
            points = [(trapezoid[i], trapezoid[i+1]) for i in range(0, 8, 2)]
            for i in range(4):
                pt1 = points[i]
                pt2 = points[(i + 1) % 4]
                cv2.line(self.rect_img, pt1, pt2, color, RECT_DRAWING_WIDTH, cv2.LINE_AA)

    def draw_trapezoid(self, trapezoid, color):
        """Function to draw a single trapezoid"""
        points = [(trapezoid[i], trapezoid[i+1]) for i in range(0, 8, 2)]
        for i in range(4):
            pt1 = points[i]
            pt2 = points[(i + 1) % 4]
            cv2.line(self.image, pt1, pt2, color, RECT_DRAWING_WIDTH, cv2.LINE_AA)

    def find_intersections(self):
        """Function to find all the intersection points between a set of lines"""
        self.intersections = []
        
        for i in range(len(self.lines)):
            for j in range(i + 1, len(self.lines)):
                intersects, intersection = self.intersect(self.lines[i], self.lines[j])
                if intersects:
                    self.intersections.append(intersection)
        
        if NMS_THRESHOLD > 1:
            self.non_maximum_suppression_points()

    def create_graph(self):
        """Function to create a graph where nodes are intersection points and edges are lines that join them"""
        self.adjacency_matrix = [[False for _ in range(len(self.intersections))] 
                                for _ in range(len(self.intersections))]
        
        for i in range(len(self.intersections)):
            for j in range(i + 1, len(self.intersections)):
                if self.is_white_line(self.intersections[i], self.intersections[j]):
                    self.adjacency_matrix[i][j] = True
                    self.adjacency_matrix[j][i] = True
        
        if DRAW_GRAPH:
            self.draw_graph_func()

    def extract_trapezoids(self):
        """Function to extract all possible trapezoids from a graph"""
        self.trapezoids = []
        self.reduce_saturation_brightness(0.5, 0.5)
        
        min_area = RECTS_MIN_AREA_FACTOR * self.image.shape[1] * self.image.shape[0]
        max_area = RECTS_MAX_AREA_FACTOR * self.image.shape[1] * self.image.shape[0]
        
        n = len(self.intersections)
        for i in range(n):
            for j in range(i + 1, n):
                if not self.adjacency_matrix[i][j]:
                    continue
                
                for k in range(i + 1, n):
                    if not self.adjacency_matrix[k][j]:
                        continue
                    
                    for l in range(j + 1, n):
                        if not self.adjacency_matrix[k][l] or not self.adjacency_matrix[i][l]:
                            continue
                        
                        points = [self.intersections[i], self.intersections[j], 
                                self.intersections[k], self.intersections[l]]
                        
                        # Test trapezoid validity
                        if not self.trapezoid_line_test(*points):
                            self.draw_trapezoid([points[0][0], points[0][1], points[1][0], points[1][1],
                                               points[2][0], points[2][1], points[3][0], points[3][1]], 
                                               (0, 0, 128))
                            continue
                        
                        if not self.trapezoid_area_test(*points, min_area, max_area):
                            self.draw_trapezoid([points[0][0], points[0][1], points[1][0], points[1][1],
                                               points[2][0], points[2][1], points[3][0], points[3][1]], 
                                               (128, 0, 0))
                            continue
                        
                        if not self.trapezoid_angles_test(*points):
                            self.draw_trapezoid([points[0][0], points[0][1], points[1][0], points[1][1],
                                               points[2][0], points[2][1], points[3][0], points[3][1]], 
                                               (0, 128, 0))
                            continue
                        
                        # Check if contour is convex
                        contour = np.array(points, dtype=np.int32)
                        if not cv2.isContourConvex(contour):
                            self.draw_trapezoid([points[0][0], points[0][1], points[1][0], points[1][1],
                                               points[2][0], points[2][1], points[3][0], points[3][1]], 
                                               (128, 128, 0))
                            continue
                        
                        # Valid trapezoid found
                        trapezoid = [points[0][0], points[0][1], points[1][0], points[1][1],
                                   points[2][0], points[2][1], points[3][0], points[3][1]]
                        self.trapezoids.append(trapezoid)
                        
                        if len(self.trapezoids) >= MAX_NUM_TRAPEZOIDS:
                            return

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LineDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



