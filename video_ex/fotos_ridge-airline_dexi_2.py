
import cv2
import numpy as np
import os
import torch
import math
from unet import UNet
from otros.deximodel import DexiNed

# === Configuración del detector de orientación ===
THETA_RESOLUTION = 6
KERNEL_SIZE = 9

# Selección del modelo (0 para DexiNed, 1 para UNet)
USING_UNET = False

# Rutas a los checkpoints del modelo
DEXINED_PATH = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/dexi.pth'
UNET_PATH = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/unet.pth'


def init_orientation_detector(theta_resolution, kernel_size, device):
    """Inicializar detector de orientación con kernels de línea"""
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
        # Crear una línea en el kernel en este ángulo
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


class RectangleDetector:
    def __init__(
        self,
        elongation_factor=1.3,
        rects_min_area_factor=0.005,
        rects_max_area_factor=0.3,
        compactness_threshold=0.5,
        length_diff_threshold=0.1,
        nms_threshold=4,
        lineImg_drawing_width=1,
        rectImg_drawing_width=2,
        draw_graph=True,
        draw_trapezoids=True,
        whiteness_threshold=0.85,
        angle_threshold=0.15,
        max_num_trapezoids=5,
    ):
        self.elongation_factor = elongation_factor
        self.rects_min_area_factor = rects_min_area_factor
        self.rects_max_area_factor = rects_max_area_factor
        self.compactness_threshold = compactness_threshold
        self.length_diff_threshold = length_diff_threshold
        self.nms_threshold = nms_threshold
        self.lineImg_drawing_width = lineImg_drawing_width
        self.rectImg_drawing_width = rectImg_drawing_width
        self.draw_graph = draw_graph
        self.draw_trapezoids = draw_trapezoids
        self.whiteness_threshold = whiteness_threshold
        self.angle_threshold = angle_threshold
        self.max_num_trapezoids = max_num_trapezoids
        
        # Configurar dispositivo
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("DISPOSITIVO: ", self.device)
        
        # Cargar modelo
        if not USING_UNET:
            self.model = DexiNed().to(self.device)
            self.model.load_state_dict(torch.load(DEXINED_PATH, map_location=self.device))
            print('Cargado modelo DexiNed')
        else:
            self.model = UNet(1,1).to(self.device)
            self.model.load_state_dict(torch.load(UNET_PATH, map_location=self.device))
            print('Cargado modelo UNet')
        
        # Preparar detector de orientación
        self.orientation_detector = init_orientation_detector(
            THETA_RESOLUTION,
            KERNEL_SIZE,
            self.device
        )

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = np.sum(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect.astype(int)

    def elongate_lines(self, lines):
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
        dist = int(np.linalg.norm(np.array(p2) - np.array(p1)))
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
        p = np.array([line1[0], line1[1]])
        r = np.array([line1[2] - line1[0], line1[3] - line1[1]])
        q = np.array([line2[0], line2[1]])
        s = np.array([line2[2] - line2[0], line2[3] - line2[1]])
        rxs = r[0] * s[1] - r[1] * s[0]
        if rxs == 0:
            return None
        t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / rxs
        u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / rxs
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p + t * r
            return (int(round(intersection[0])), int(round(intersection[1])))
        return None

    def trapezoid_line_test(self, a, b, c, d):
        lengths = [
            np.linalg.norm(np.array(a) - np.array(b)),
            np.linalg.norm(np.array(b) - np.array(c)),
            np.linalg.norm(np.array(c) - np.array(d)),
            np.linalg.norm(np.array(d) - np.array(a)),
        ]
        min_len, max_len = min(lengths), max(lengths)
        return min_len / max_len >= self.length_diff_threshold

    def trapezoid_area_test(self, a, b, c, d, min_area, max_area):
        pts = np.array([a, b, c, d])
        area = 0.5 * abs(
            np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
            - np.dot(pts[:, 1], np.roll(pts[:, 0], 1))
        )
        if area < min_area or area > max_area:
            return False
        perimeter = sum(np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4))
        pp = 4 * np.pi * area / (perimeter * perimeter)
        return pp >= self.compactness_threshold

    def trapezoid_angles_test(self, a, b, c, d):
        def angle(pt1, pt2, pt3):
            v1 = np.array(pt1) - np.array(pt2)
            v2 = np.array(pt3) - np.array(pt2)
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.arccos(np.clip(cos_angle, -1.0, 1.0))

        pts = [a, b, c, d]
        for i in range(4):
            ang = angle(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4])
            if not (
                np.pi / 2 - self.angle_threshold
                <= ang
                <= np.pi / 2 + self.angle_threshold
            ):
                return False
        return True

    def is_rectangle(self, pts):
        return self.trapezoid_angles_test(*pts)

    def non_maximum_suppression_points(self, points):
        suppressed = [False] * len(points)
        filtered = []
        for i in range(len(points)):
            if suppressed[i]:
                continue
            close_pts = [points[i]]
            for j in range(i + 1, len(points)):
                if (
                    np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                    < self.nms_threshold
                ):
                    suppressed[j] = True
                    close_pts.append(points[j])
            avg_pt = np.mean(close_pts, axis=0)
            filtered.append((int(avg_pt[0]), int(avg_pt[1])))
        return filtered

    def extract_lines(self, image):
        """Extraer líneas usando el modelo DexiNed/UNet en lugar de FLD"""
        # Redimensionar a múltiplo de 16
        h, w = image.shape[:2]
        res = 16
        new_w = (w // res) * res
        new_h = (h // res) * res
        img = cv2.resize(image, (new_w, new_h))

        # Convertir a entrada del modelo
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

        # Detección de líneas con Hough
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=15,
            maxLineGap=5
        )

        if lines is None:
            return []
        
        # Convertir el formato de las líneas
        lines_list = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            lines_list.append([x1, y1, x2, y2])
        
        return self.elongate_lines(lines_list)

    def find_intersections(self, lines):
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self.intersect(lines[i], lines[j])
                if pt:
                    intersections.append(pt)
        return self.non_maximum_suppression_points(intersections)

    def create_graph(self, intersections, lines, lines_img):
        size = len(intersections)
        adj = [[False] * size for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                if self.is_white_line(lines_img, intersections[i], intersections[j]):
                    adj[i][j] = adj[j][i] = True
        return adj

    def extract_trapezoids(
        self, intersections, adjacency, image, debug_img=None, graph_black=None
    ):
        if debug_img is None:
            debug_img = image.copy()

        trapezoids = []
        failed_angle_traps = []
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

                        if not self.trapezoid_line_test(*pts):
                            self.draw_quadrilaterals(debug_img, [pts], (0, 0, 128))
                            if graph_black is not None:
                                self.draw_quadrilaterals(
                                    graph_black, [pts], (0, 0, 128)
                                )
                            continue
                        if not self.trapezoid_area_test(*pts, min_area, max_area):
                            self.draw_quadrilaterals(debug_img, [pts], (128, 0, 0))
                            if graph_black is not None:
                                self.draw_quadrilaterals(
                                    graph_black, [pts], (128, 0, 0)
                                )
                            continue
                        if not self.trapezoid_angles_test(*pts):
                            self.draw_quadrilaterals(debug_img, [pts], (0, 128, 0))
                            if graph_black is not None:
                                self.draw_quadrilaterals(
                                    graph_black, [pts], (0, 128, 0)
                                )
                            failed_angle_traps.append(pts)
                            continue
                        if not cv2.isContourConvex(contour):
                            self.draw_quadrilaterals(debug_img, [pts], (0, 128, 128))
                            if graph_black is not None:
                                self.draw_quadrilaterals(
                                    graph_black, [pts], (0, 128, 128)
                                )
                            continue

                        trapezoids.append(pts)
                        if len(trapezoids) >= self.max_num_trapezoids:
                            return trapezoids, failed_angle_traps

        cv2.imwrite("debug_trapezoids.png", debug_img)
        return trapezoids, failed_angle_traps

    def draw_quadrilaterals(self, img, quads, color):
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

    def reduce_saturation_brightness(
        self, image, saturation_scale=0.5, value_scale=0.5
    ):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s *= saturation_scale
        v *= value_scale
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def process_image(self, image, output_folder, base_name):
        image = self.reduce_saturation_brightness(image, 0.5, 0.5)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines_img = np.zeros_like(gray)
        lines = self.extract_lines(image)

        for line in lines:
            cv2.line(
                lines_img,
                (int(line[0]), int(line[1])),
                (int(line[2]), int(line[3])),
                255,
                self.lineImg_drawing_width,
                cv2.LINE_AA,
            )

        lines_img_blur = cv2.GaussianBlur(lines_img, (5, 5), 0)

        intersections = self.find_intersections(lines)
        adjacency = self.create_graph(intersections, lines, lines_img_blur)

        # Crear imagen sobre fondo negro para graficar después
        graph_black = np.zeros_like(image)

        # Dibujar grafo si está activado
        graph_img = image.copy()
        if self.draw_graph:
            for i in range(len(intersections)):
                for j in range(i + 1, len(intersections)):
                    if adjacency[i][j]:
                        # Líneas magenta en graph_img
                        cv2.line(
                            graph_img,
                            intersections[i],
                            intersections[j],
                            (255, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        # Líneas magenta en graph_black
                        cv2.line(
                            graph_black,
                            intersections[i],
                            intersections[j],
                            (255, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )

            # Dibujar puntos blancos en graph_black
            for pt in intersections:
                cv2.circle(graph_black, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

        debug_img = image.copy()
        trapezoids, failed_angle_traps = self.extract_trapezoids(
            intersections,
            adjacency,
            image,
            debug_img=debug_img,
            graph_black=graph_black,
        )

        rectangles, others = [], []
        for t in trapezoids:
            (rectangles if self.is_rectangle(t) else others).append(t)

        output = image.copy()
        self.draw_quadrilaterals(output, rectangles, (255, 255, 255))  # Blanco
        if self.draw_trapezoids:
            self.draw_quadrilaterals(output, others, (0, 255, 255))  # Amarillo claro
        
        image_with_failed_angles = image.copy()
        self.draw_quadrilaterals(
            image_with_failed_angles, rectangles, (255, 255, 255)
        )  # rectángulos en blanco
        self.draw_quadrilaterals(
            image_with_failed_angles, failed_angle_traps, (255, 255, 255)
        )  # trapecios que fallan ángulos en blanco
        
        # Guardar imágenes de depuración
        cv2.imwrite(
            os.path.join(output_folder, f"{base_name}_debug_lines_blur.png"),
            lines_img_blur,
        )
        cv2.imwrite(
            os.path.join(output_folder, f"{base_name}_debug_output.png"), output
        )
        cv2.imwrite(
            os.path.join(output_folder, f"{base_name}_debug_graph_black.png"),
            graph_black,
        )
        cv2.imwrite(
            os.path.join(output_folder, f"{base_name}_rects_plus_failed_angles.png"),
            image_with_failed_angles,
        )

        return output, rectangles, others


def main(input_folder, output_folder):
    detector = RectangleDetector()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            print(f"Procesando {filename}...")
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error al leer {image_path}")
                continue

            base_name = os.path.splitext(filename)[0]
            output_img, rects, traps = detector.process_image(
                image, output_folder, base_name
            )
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, output_img)

            print(f"Guardado: {output_path}")
            print(f"Rectángulos: {len(rects)}, Otros trapecios: {len(traps)}\n")


if __name__ == '__main__':
    input_folder = "/home/zzh/Robot_Home"
    output_folder = "/home/zzh/Robot_Home/RIDGE-DEXINED"

    main(input_folder, output_folder)