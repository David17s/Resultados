#!/usr/bin/env python3
import os
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
import timeit
import math
from itertools import combinations
from unet import UNet
from otros.deximodel import DexiNed


# === Orientation detector settings ===
THETA_RESOLUTION = 6
KERNEL_SIZE = 9

# Model selection (0 for DexiNed, 1 for UNet)
USING_UNET = False

# Paths to your model checkpoints
DEXINED_PATH = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/dexi.pth'
UNET_PATH = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/unet.pth'

# Rectangle detection parameters
ELONGATION_FACTOR = 1.3
RECTS_MIN_AREA_FACTOR = 0.005
RECTS_MAX_AREA_FACTOR = 0.3
COMPACTNESS_THRESHOLD = 0.5
LENGTH_DIFF_THRESHOLD = 0.1
NMS_THRESHOLD = 4
LINE_DRAWING_WIDTH = 1
RECT_DRAWING_WIDTH = 2
DRAW_GRAPH = True
DRAW_TRAPEZOIDS = True
WHITENESS_THRESHOLD = 0.85
ANGLE_THRESHOLD = 0.15
EXECUTION_PERIOD = 0.1
MAX_NUM_TRAPEZOIDS = 5


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


class NeuralRIDGEDetector:
    def __init__(self):
        super().__init__()
        
        # Setup device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("DISPOSITIVO: ", self.device)

        # Load neural network model
        if not USING_UNET:
            self.model = DexiNed().to(self.device)
            self.model.load_state_dict(torch.load(DEXINED_PATH, map_location=self.device))
            print('Loaded DexiNed model')
        else:
            self.model = UNet(1,1).to(self.device)
            self.model.load_state_dict(torch.load(UNET_PATH, map_location=self.device))
            print('Loaded UNet model')
        
        # Prepare orientation detector
        self.orientation_detector = init_orientation_detector(
            THETA_RESOLUTION,
            KERNEL_SIZE,
            self.device
        )

        # Initialize RIDGE detector parameters
        self.elongation_factor = ELONGATION_FACTOR
        self.rects_min_area_factor = RECTS_MIN_AREA_FACTOR
        self.rects_max_area_factor = RECTS_MAX_AREA_FACTOR
        self.compactness_threshold = COMPACTNESS_THRESHOLD
        self.length_diff_threshold = LENGTH_DIFF_THRESHOLD
        self.nms_threshold = NMS_THRESHOLD
        self.lineImg_drawing_width = LINE_DRAWING_WIDTH
        self.rectImg_drawing_width = RECT_DRAWING_WIDTH
        self.draw_graph = DRAW_GRAPH
        self.draw_trapezoids = DRAW_TRAPEZOIDS
        self.whiteness_threshold = WHITENESS_THRESHOLD
        self.angle_threshold = ANGLE_THRESHOLD
        self.max_num_trapezoids = MAX_NUM_TRAPEZOIDS

        # Initialize variables for rectangle detection
        self.lines = []
        self.intersections = []
        self.trapezoids = []
        self.adjacency_matrix = []
        self.image = None
        self.gray = None
        self.lines_img = None
        self.graph_img = None

        # Process images
        input_folder = "/home/zzh/Robot_Home"
        output_folder = "/home/zzh/Robot_Home/RIDGE-AIRLINE-DEXI"



        try:
            # Crear carpeta de salida si no existe
            os.makedirs(output_folder, exist_ok=True)
            
            # Extensiones de imagen soportadas
            extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            imagenes = []

            for ext in extensiones:
                imagenes.extend(glob.glob(os.path.join(input_folder, ext)))
                imagenes.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
            if not imagenes:
                print("No se encontraron imágenes en la carpeta: "+ str(input_folder))
                return
        
            print(f"Procesando {len(imagenes)} imágenes...")

            # Procesar cada imagen
            for i, imagen_path in enumerate(imagenes):
                try:
                    # Leer imagen
                    frame = cv2.imread(imagen_path)
                    if frame is None:
                        print("No se pudo cargar la imagen: " + str(imagen_path))
                        continue
                    
                    # Obtener nombre del archivo
                    filename = os.path.basename(imagen_path)
                    
                    # Procesar imagen
                    self.process_image_neural_ridge(frame, output_folder, filename)
                    
                    print(f"Procesada {i+1}/{len(imagenes)}: {filename}")
                    
                except Exception as e:
                    print("Error procesando " + str(imagen_path) + ": " + str(e))
                    continue
            
            print(f"Procesamiento completado. Resultados guardados en: {output_folder}")
    
        except Exception as e:
            print("Error procesando carpeta: " + str(e))

    def detect_lines_neural(self, image):
        """Detect lines using neural network (DexiNed or UNet)"""
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

        # Convert lines to RIDGE format
        ridge_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                ridge_lines.append([x1, y1, x2, y2])

        return ridge_lines, edges, img

    # RIDGE Rectangle Detection Methods
    def order_points_clockwise(self, pts):
        """Order points in clockwise manner"""
        rect = np.zeros((4, 2), dtype="float32")
        s = np.sum(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect.astype(int)

    def elongate_lines(self, lines):
        """Elongate lines by elongation factor"""
        extended_lines = []
        extension = (self.elongation_factor - 1) / 2
        for line in lines:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1
            new_x1 = x1 - dx * extension
            new_y1 = y1 - dy * extension
            new_x2 = x2 + dx * extension
            new_y2 = y2 + dy * extension
            extended_lines.append([new_x1, new_y1, new_x2, new_y2])
        return extended_lines

    def is_white_line(self, img, p1, p2):
        """Check if line is sufficiently white in the line image"""
        dist = int(np.linalg.norm(np.array(p2) - np.array(p1)))
        if dist == 0:
            return False
        
        white_count, total = 0, 0
        for t in np.linspace(0, 1, dist):
            x = int(round(p1[0] + t * (p2[0] - p1[0])))
            y = int(round(p1[1] + t * (p2[1] - p1[1])))
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                if img[y, x] > 64:
                    white_count += 1
                total += 1
        return total > 0 and white_count / total >= self.whiteness_threshold

    def intersect(self, line1, line2):
        """Find intersection point between two lines"""
        p = np.array([line1[0], line1[1]])
        r = np.array([line1[2] - line1[0], line1[3] - line1[1]])
        q = np.array([line2[0], line2[1]])
        s = np.array([line2[2] - line2[0], line2[3] - line2[1]])
        
        rxs = r[0] * s[1] - r[1] * s[0]
        if abs(rxs) < 1e-10:
            return None
        
        qp = q - p
        t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
        u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p + t * r
            return (int(round(intersection[0])), int(round(intersection[1])))
        return None

    def trapezoid_line_test(self, a, b, c, d):
        """Test if trapezoid has acceptable side length ratios"""
        lengths = [
            np.linalg.norm(np.array(a) - np.array(b)),
            np.linalg.norm(np.array(b) - np.array(c)),
            np.linalg.norm(np.array(c) - np.array(d)),
            np.linalg.norm(np.array(d) - np.array(a)),
        ]
        min_len, max_len = min(lengths), max(lengths)
        return min_len / max_len >= self.length_diff_threshold

    def trapezoid_area_test(self, a, b, c, d, min_area, max_area):
        """Test if trapezoid has acceptable area and compactness"""
        pts = np.array([a, b, c, d])
        area = 0.5 * abs(
            np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
            - np.dot(pts[:, 1], np.roll(pts[:, 0], 1))
        )
        if area < min_area or area > max_area:
            return False
        
        perimeter = sum(np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4))
        if perimeter == 0:
            return False
        
        pp = 4 * np.pi * area / (perimeter * perimeter)
        return pp >= self.compactness_threshold

    def trapezoid_angles_test(self, a, b, c, d):
        """Test if trapezoid has angles close to 90 degrees"""
        def angle(pt1, pt2, pt3):
            v1 = np.array(pt1) - np.array(pt2)
            v2 = np.array(pt3) - np.array(pt2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                return 0
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            return np.arccos(np.clip(cos_angle, -1.0, 1.0))

        pts = [a, b, c, d]
        for i in range(4):
            ang = angle(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4])
            if not (np.pi / 2 - self.angle_threshold <= ang <= np.pi / 2 + self.angle_threshold):
                return False
        return True

    def is_rectangle(self, pts):
        """Check if the quadrilateral is a rectangle"""
        return self.trapezoid_angles_test(*pts)

    def non_maximum_suppression_points(self, points):
        """Remove close intersection points"""
        if not points:
            return []
        
        suppressed = [False] * len(points)
        filtered = []
        
        for i in range(len(points)):
            if suppressed[i]:
                continue
            close_pts = [points[i]]
            for j in range(i + 1, len(points)):
                if (np.linalg.norm(np.array(points[i]) - np.array(points[j])) < self.nms_threshold):
                    suppressed[j] = True
                    close_pts.append(points[j])
            
            avg_pt = np.mean(close_pts, axis=0)
            filtered.append((int(avg_pt[0]), int(avg_pt[1])))
        
        return filtered

    def find_intersections(self, lines):
        """Find all intersection points between lines"""
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self.intersect(lines[i], lines[j])
                if pt:
                    intersections.append(pt)
        return self.non_maximum_suppression_points(intersections)

    def create_graph(self, intersections, lines_img):
        """Create adjacency graph between intersection points"""
        size = len(intersections)
        adj = [[False] * size for _ in range(size)]
        
        for i in range(size):
            for j in range(i + 1, size):
                if self.is_white_line(lines_img, intersections[i], intersections[j]):
                    adj[i][j] = adj[j][i] = True
        
        return adj

    def extract_trapezoids(self, intersections, adjacency, image):
        """Extract trapezoids from the intersection graph"""
        trapezoids = []
        min_area = self.rects_min_area_factor * image.shape[1] * image.shape[0]
        max_area = self.rects_max_area_factor * image.shape[1] * image.shape[0]
        n = len(intersections)

        for i in range(n):
            for j in range(i + 1, n):
                if not adjacency[i][j]:
                    continue
                for k in range(i + 1, n):
                    if not adjacency[k][j]:
                        continue
                    for l in range(j + 1, n):
                        if not (adjacency[k][l] and adjacency[i][l]):
                            continue
                        
                        raw_pts = np.array([intersections[x] for x in [i, j, k, l]])
                        pts = self.order_points_clockwise(raw_pts)
                        contour = np.array(pts)
                        
                        # Apply all tests
                        if not self.trapezoid_line_test(*pts):
                            continue
                        if not self.trapezoid_area_test(*pts, min_area, max_area):
                            continue
                        if not self.trapezoid_angles_test(*pts):
                            continue
                        if not cv2.isContourConvex(contour):
                            continue
                        
                        trapezoids.append(pts)
                        if len(trapezoids) >= self.max_num_trapezoids:
                            return trapezoids
        
        return trapezoids

    def draw_quadrilaterals(self, img, quads, color):
        """Draw quadrilaterals on image"""
        for quad in quads:
            for i in range(4):
                cv2.line(
                    img,
                    tuple(quad[i]),
                    tuple(quad[(i + 1) % 4]),
                    color,
                    self.rectImg_drawing_width,
                    cv2.LINE_AA,
                )

    def reduce_saturation_brightness(self, image, saturation_scale=0.5, value_scale=0.5):
        """Reduce saturation and brightness of image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s *= saturation_scale
        v *= value_scale
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def process_image_neural_ridge(self, image, output_folder, filename):
        """Complete pipeline: Neural network line detection + RIDGE rectangle detection"""
        try:
            # Step 1: Detect lines using neural network
            neural_lines, edges, processed_img = self.detect_lines_neural(image)
            
            if not neural_lines:
                print(f"No lines detected for {filename}")
                return
            
            # Step 1.5: Create AIRLINE visualization (neural network lines in yellow)
            airline_img = processed_img.copy()
            for line in neural_lines:
                cv2.line(
                    airline_img,
                    (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])),
                    (0, 255, 255),  # Yellow color for AIRLINE
                    2,
                    cv2.LINE_AA,
                )
            
            # Step 2: Prepare image for RIDGE processing
            processed_img = self.reduce_saturation_brightness(processed_img, 0.5, 0.5)
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            
            # Step 3: Elongate lines (RIDGE preprocessing)
            extended_lines = self.elongate_lines(neural_lines)
            
            # Step 4: Create lines image
            lines_img = np.zeros_like(gray)
            for line in extended_lines:
                cv2.line(
                    lines_img,
                    (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])),
                    255,
                    self.lineImg_drawing_width,
                    cv2.LINE_AA,
                )
            
            lines_img_blur = cv2.GaussianBlur(lines_img, (5, 5), 0)
            
            # Step 5: Find intersections
            intersections = self.find_intersections(extended_lines)
            
            if len(intersections) < 4:
                print(f"Not enough intersections for {filename}")
                return
            
            # Step 6: Create adjacency graph
            adjacency = self.create_graph(intersections, lines_img_blur)
            
            # Step 7: Extract trapezoids
            trapezoids = self.extract_trapezoids(intersections, adjacency, processed_img)
            
            # Step 8: Classify rectangles vs other trapezoids
            rectangles, others = [], []
            for t in trapezoids:
                (rectangles if self.is_rectangle(t) else others).append(t)
            
            # Step 9: Draw results
            output = processed_img.copy()
            self.draw_quadrilaterals(output, rectangles, (255, 255, 255))  # White rectangles
            if self.draw_trapezoids:
                self.draw_quadrilaterals(output, others, (0, 255, 255))  # Yellow trapezoids
            
            # Step 10: Save results
            base_name = os.path.splitext(filename)[0]
            
            # Save all intermediate results
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_original.jpg"), image)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_edges.jpg"), edges)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_airline.jpg"), airline_img)  # NEW: AIRLINE lines in yellow
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_lines.jpg"), lines_img)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_lines_blur.jpg"), lines_img_blur)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_final_result.jpg"), output)
            
            print(f"  - AIRLINE lines detected: {len(neural_lines)}")
            print(f"  - Intersections: {len(intersections)}")
            print(f"  - Rectangles: {len(rectangles)}")
            print(f"  - Other trapezoids: {len(others)}")
            
        except Exception as e:
            print(f'Error processing {filename}: {str(e)}')

if __name__ == '__main__':
    detector = NeuralRIDGEDetector()
    
    print("Presiona cualquier tecla para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()