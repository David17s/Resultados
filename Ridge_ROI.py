import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

from scipy.spatial.distance import cdist
import math


def calculate_iou(rect1, rect2):
    """
    Calcula el Intersection over Union (IoU) entre dos rectángulos.
    rect1, rect2: (x, y, w, h)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Coordenadas de intersección
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Si no hay intersección
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Área de intersección
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Área de unión
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def calculate_overlap_percentage(rect1, rect2):
    """
    Calcula el porcentaje de superposición del rectángulo más pequeño.
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Coordenadas de intersección
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Si no hay intersección
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Área de intersección
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Área del rectángulo más pequeño
    area1 = w1 * h1
    area2 = w2 * h2
    smaller_area = min(area1, area2)

    return intersection_area / smaller_area if smaller_area > 0 else 0.0


def filter_overlapping_rectangles(
    rectangulos_info, iou_threshold=0.3, overlap_threshold=0.5
):
    """
    Filtra rectángulos superpuestos usando IoU y porcentaje de superposición.
    """
    if not rectangulos_info:
        return []

    # Ordenar por área (los más grandes primero)
    rectangulos_sorted = sorted(rectangulos_info, key=lambda x: x["area"], reverse=True)

    filtered_rectangles = []

    for i, rect_current in enumerate(rectangulos_sorted):
        should_keep = True
        current_coords = rect_current["coordenadas_originales"]

        for rect_filtered in filtered_rectangles:
            filtered_coords = rect_filtered["coordenadas_originales"]

            # Calcular IoU
            iou = calculate_iou(current_coords, filtered_coords)

            # Calcular porcentaje de superposición
            overlap_pct = calculate_overlap_percentage(current_coords, filtered_coords)

            # Si hay mucha superposición, descartar el rectángulo actual
            if iou > iou_threshold or overlap_pct > overlap_threshold:
                should_keep = False
                break

        if should_keep:
            filtered_rectangles.append(rect_current)

    return filtered_rectangles


def non_max_suppression_rectangles(rectangulos_info, overlap_threshold=0.3):
    """
    Implementa Non-Maximum Suppression para rectángulos.
    """
    if not rectangulos_info:
        return []

    # Convertir a formato para NMS
    boxes = []
    scores = []

    for rect in rectangulos_info:
        x, y, w, h = rect["coordenadas_originales"]
        boxes.append([x, y, x + w, y + h])
        scores.append(rect["area"])  # Usar área como "score"

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    # Aplicar NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, overlap_threshold)

    # Filtrar rectángulos
    filtered_rectangles = []
    if len(indices) > 0:
        for i in indices.flatten():
            filtered_rectangles.append(rectangulos_info[i])

    return filtered_rectangles


class RectangleDetector:
    def __init__(
        self,
        elongation_factor=1.3,
        rects_min_area_factor=0.001,
        rects_max_area_factor=0.3,
        compactness_threshold=0.3,
        length_diff_threshold=0.1,
        nms_threshold=10,
        lineImg_drawing_width=1,
        rectImg_drawing_width=2,
        draw_graph=True,
        draw_trapezoids=True,
        whiteness_threshold=0.85,
        angle_threshold=0.15,
        max_num_trapezoids=7,
        iou_threshold=0.3,  # IoU threshold for shape NMS
    ):
        self.elongation_factor = elongation_factor
        self.rects_min_area_factor = rects_min_area_factor
        self.rects_max_area_factor = rects_max_area_factor
        self.compactness_threshold = compactness_threshold
        self.length_diff_threshold = length_diff_threshold
        self.nms_threshold = nms_threshold  # For intersection points
        self.lineImg_drawing_width = lineImg_drawing_width
        self.rectImg_drawing_width = rectImg_drawing_width
        self.draw_graph = (draw_graph,)
        self.draw_trapezoids = draw_trapezoids
        self.whiteness_threshold = whiteness_threshold
        self.angle_threshold = angle_threshold
        self.max_num_trapezoids = max_num_trapezoids
        self.iou_threshold = iou_threshold  # For shape NMS

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
        p = np.array([line1[0], line1[1]])
        r = np.array([line1[2] - line1[0], line1[3] - line1[1]])
        q = np.array([line2[0], line2[1]])
        s = np.array([line2[2] - line2[0], line2[3] - line2[1]])
        rxs = r[0] * s[1] - r[1] * s[0]
        if rxs == 0:  # Parallel or collinear
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
        min_len = min(lengths)
        max_len = max(lengths)
        return min_len / max_len >= self.length_diff_threshold if max_len > 0 else False

    def trapezoid_area_test(self, a, b, c, d, min_area, max_area):
        pts = np.array([a, b, c, d])
        area = 0.5 * abs(
            np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
            - np.dot(pts[:, 1], np.roll(pts[:, 0], 1))
        )
        if area < min_area or area > max_area:
            return False
        perimeter = sum(np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4))
        if perimeter == 0:  # Avoid division by zero for degenerate shapes
            return False
        pp = 4 * np.pi * area / (perimeter * perimeter)
        return pp >= self.compactness_threshold

    def trapezoid_angles_test(self, a, b, c, d):
        def angle(pt1, pt2, pt3):
            v1 = np.array(pt1) - np.array(pt2)
            v2 = np.array(pt3) - np.array(pt2)

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:  # Handle zero-length vectors
                return 0

            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
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
        if not points:
            return []

        # Convert to numpy array for easier calculations
        points_np = np.array(points)

        # Sort points by x-coordinate (or any other order)
        # For simplicity, let's just use the order they come in,
        # as the filtering is based on proximity, not confidence.

        filtered_points = []
        suppressed = [False] * len(points_np)

        for i in range(len(points_np)):
            if suppressed[i]:
                continue

            current_point = points_np[i]
            close_pts = [current_point]

            for j in range(i + 1, len(points_np)):
                if suppressed[j]:
                    continue

                distance = np.linalg.norm(current_point - points_np[j])
                if distance < self.nms_threshold:
                    suppressed[j] = True
                    close_pts.append(points_np[j])

            avg_pt = np.mean(close_pts, axis=0)
            filtered_points.append((int(avg_pt[0]), int(avg_pt[1])))

        return filtered_points

    def extract_lines(self, gray):
        otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_th1 = float(0.33 * otsu_val)
        canny_th2 = float(otsu_val)
        fld = cv2.ximgproc.createFastLineDetector(
            length_threshold=15,
            distance_threshold=1.414,
            canny_th1=canny_th1,
            canny_th2=canny_th2,
            canny_aperture_size=3,
            do_merge=True,
        )
        lines = fld.detect(gray)
        if lines is None:
            return []
        lines = [line[0].tolist() for line in lines]
        return self.elongate_lines(lines)

    def find_intersections(self, lines):
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self.intersect(lines[i], lines[j])
                if pt:
                    intersections.append(pt)
        return self.non_maximum_suppression_points(intersections)

    def create_graph(self, intersections, lines_img):
        size = len(intersections)
        adj = [[False] * size for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                if self.is_white_line(lines_img, intersections[i], intersections[j]):
                    adj[i][j] = adj[j][i] = True
        return adj

    def extract_trapezoids(self, intersections, adjacency, image):
        trapezoids = []
        min_area = self.rects_min_area_factor * image.shape[1] * image.shape[0]
        max_area = self.rects_max_area_factor * image.shape[1] * image.shape[0]
        n = len(intersections)

        for i in range(n):
            for j in range(i + 1, n):
                if not adjacency[i][j]:
                    continue
                for k in range(i + 1, n):
                    if not adjacency[k][j]:  # Ensure k is connected to j
                        continue
                    for l in range(j + 1, n):  # Ensure l is distinct from i,j,k
                        if not (
                            adjacency[k][l] and adjacency[i][l]
                        ):  # Ensure l connected to k and i
                            continue

                        raw_pts = np.array([intersections[x] for x in [i, j, k, l]])
                        pts = self.order_points_clockwise(raw_pts)
                        contour = np.array(pts)

                        # Apply all trapezoid tests
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

    def non_maximum_suppression_shapes(self, shapes):
        if not shapes:
            return []

        # Convert shapes to bounding boxes for IoU calculation
        bboxes = []
        for shape in shapes:
            x_coords = [p[0] for p in shape]
            y_coords = [p[1] for p in shape]
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)
            bboxes.append([x1, y1, x2, y2])

        # Perform NMS on bounding boxes
        indices = cv2.dnn.NMSBoxes(bboxes, [1.0] * len(bboxes), 0.5, self.iou_threshold)

        if len(indices) == 0:
            return []

        # indices are returned as [[idx1], [idx2], ...]
        if isinstance(indices[0], (list, np.ndarray)):
            indices = [i[0] for i in indices]

        filtered_shapes = [shapes[i] for i in indices]
        return filtered_shapes

    def process_roi(self, roi_image):
        gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        lines_img = np.zeros_like(gray_roi)
        lines = self.extract_lines(gray_roi)

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
        adjacency = self.create_graph(intersections, lines_img_blur)

        trapezoids = self.extract_trapezoids(intersections, adjacency, roi_image)

        rectangles = []
        other_trapezoids = []
        for t in trapezoids:
            if self.is_rectangle(t):
                rectangles.append(t)
            else:
                other_trapezoids.append(t)

        return lines, intersections, rectangles, other_trapezoids


def reduce_saturation_brightness(image, saturation_scale=0.5, value_scale=0.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s *= saturation_scale
    v *= value_scale
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def detectar_rectangulos(image, expansion=10):

    image = reduce_saturation_brightness(image, 0.8, 0.8)

    # Obtener dimensiones de la imagen
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 10, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()

    output_filtrada = image.copy()

    black_img = np.zeros_like(image)
    black2_img = image.copy()
    lines_img = np.zeros_like(gray)

    detector = RectangleDetector()

    lines = detector.extract_lines(gray)

    for line in lines:
        cv2.line(
            lines_img,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            255,
            1,
            cv2.LINE_AA,
        )

    lines_img_blur = cv2.GaussianBlur(lines_img, (5, 5), 0)

    intersections = detector.find_intersections(lines)
    adjacency = detector.create_graph(intersections, lines_img_blur)

    # Crear imagen sobre fondo negro para graficar después
    graph_black = np.zeros_like(image)

    # Dibujar grafo si está activado
    """
    
    if detector.draw_graph:
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                if adjacency[i][j]:
                    # Líneas magenta en graph_img (grosor 1, sin especificar AA, pero puedes agregarlo si quieres)
                    cv2.line(
                        black2_img,
                        intersections[i],
                        intersections[j],
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # Líneas magenta en graph_black, grosor 1 y antialiasing
                    cv2.line(
                        black2_img,
                        intersections[i],
                        intersections[j],
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # Dibujar puntos blancos en graph_black
        for pt in intersections:
            cv2.circle(black2_img, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
    
    """

    # Lista para almacenar información de los rectángulos
    rectangulos_info = []

    for i, cnt in enumerate(contours):
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)

            if 500 < area < 50000:

                # Obtener bounding box del contorno
                x, y, w, h = cv2.boundingRect(approx)

                # Expandir el bounding box
                x_exp = max(0, x - expansion)
                y_exp = max(0, y - expansion)
                w_exp = min(width - x_exp, w + 2 * expansion)
                h_exp = min(height - y_exp, h + 2 * expansion)

                # Calcular los vértices originales
                vertices_originales = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

                # Calcular los vértices expandidos
                vertices_expandidos = [
                    (x_exp, y_exp),
                    (x_exp + w_exp, y_exp),
                    (x_exp + w_exp, y_exp + h_exp),
                    (x_exp, y_exp + h_exp),
                ]

                # Dibujar los puntos originales en azul
                for vx, vy in vertices_originales:
                    cv2.circle(black_img, (vx, vy), 2, (255, 255, 255), -1)  # Azul BGR

                # Dibujar los puntos expandidos en amarillo
                for vx, vy in vertices_expandidos:
                    cv2.circle(
                        black_img, (vx, vy), 2, (255, 255, 255), -1
                    )  # Amarillo BGR

                # Obtener los vértices del contorno original
                vertices = approx.reshape(-1, 2)

                # Guardar información del rectángulo
                rect_info = {
                    "id": i,
                    "coordenadas_originales": (x, y, w, h),
                    "coordenadas_expandidas": (x_exp, y_exp, w_exp, h_exp),
                    "area": area,
                    "vertices": vertices.tolist(),
                    "contorno": approx,
                }
                rectangulos_info.append(rect_info)

                # Dibujar el contorno original en verde
                # cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)

                # Dibujar el bounding box expandido en rojo
                cv2.rectangle(
                    output,
                    (x_exp, y_exp),
                    (x_exp + w_exp, y_exp + h_exp),
                    (0, 0, 255),
                    1,
                )

                # *** NUEVA FUNCIONALIDAD: Marcar los vértices del contorno original ***
                # Dibujar círculos blancos en cada vértice del contorno
                for vertex in vertices:
                    punto = tuple(vertex.astype(int))
                    # Círculo blanco con borde negro para mejor visibilidad
                    # cv2.circle(
                    #    output, punto, 4, (255, 255, 255), -1
                    # )  # Círculo blanco relleno
                    # cv2.circle(output, punto, 4, (0, 0, 0), 2)  # Borde negro

                    """
                   
                    # Opcional: añadir coordenadas del vértice
                    cv2.putText(
                        output,
                        f"({punto[0]},{punto[1]})",
                        (punto[0] + 10, punto[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    """

                # Añadir texto con el ID del rectángulo
                cv2.putText(
                    output,
                    f"ID: {i}",
                    (x_exp, y_exp - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

    # APLICAR FILTRO DE SUPERPOSICIÓN
    # Método 1: Filtro personalizado con IoU
    rectangulos_filtrados = filter_overlapping_rectangles(
        rectangulos_info, iou_threshold=0.3, overlap_threshold=0.5
    )

    # Método 2: Non-Maximum Suppression (alternativa)
    # rectangulos_filtrados = non_max_suppression_rectangles(rectangulos_info, overlap_threshold=0.3)

    # Dibujar solo los rectángulos filtrados
    for rect_info in rectangulos_filtrados:
        approx = np.array(rect_info["contorno"])
        vertices = np.array(rect_info["vertices"])
        x, y, w, h = rect_info["coordenadas_originales"]
        x_exp, y_exp, w_exp, h_exp = rect_info["coordenadas_expandidas"]

        # Dibujar puntos originales
        vertices_originales = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        # Dibujar puntos expandidos
        vertices_expandidos = [
            (x_exp, y_exp),
            (x_exp + w_exp, y_exp),
            (x_exp + w_exp, y_exp + h_exp),
            (x_exp, y_exp + h_exp),
        ]
        for vx, vy in vertices_expandidos:
            cv2.circle(black_img, (vx, vy), 2, (255, 255, 255), -1)

        # Obtener los vértices del contorno original
        vertices = approx.reshape(-1, 2)

        # cv2.drawContours(output_filtrada, [approx], 0, (0, 255, 0), 2)

        # Dibujar bounding box expandido en rojo
        # cv2.rectangle(
        #    output_filtrada,
        #    (x_exp, y_exp),
        #    (x_exp + w_exp, y_exp + h_exp),
        #    (0, 0, 255),
        #    1,
        # )

        # Calcular el punto centro de la figura
        centro_x = int(np.mean(vertices[:, 0]))
        centro_y = int(np.mean(vertices[:, 1]))
        punto_centro = (centro_x, centro_y)

        # Dibujar el punto centro
        # cv2.circle(output_filtrada, punto_centro, 6, (255, 0, 255), -1)  # Magenta
        # cv2.circle(output_filtrada, punto_centro, 6, (0, 0, 0), 2)  # Borde negro

        for i, vertex in enumerate(vertices):
            punto = tuple(vertex.astype(int))

            # Dibujar el vértice
            cv2.circle(output_filtrada, punto, 4, (255, 255, 255), -1)
            cv2.circle(output_filtrada, punto, 4, (0, 0, 0), 2)

        # Dibujar líneas blancas entre cada par de puntos consecutivos
        for i in range(len(vertices)):
            pt1 = tuple(vertices[i].astype(int))
            pt2 = tuple(
                vertices[(i + 1) % len(vertices)].astype(int)
            )  # el % cierra el polígono
            cv2.line(output_filtrada, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

    all_detected_fld_rects = []
    all_detected_fld_trapezoids = []
    all_fld_lines = []  # To draw all FLD lines found in ROIs
    all_fld_intersections = []  # To draw all FLD intersections found in ROIs

    # Draw original detected bounding boxes
    for rect_info in rectangulos_info:
        x_orig, y_orig, w_orig, h_orig = rect_info["coordenadas_originales"]
        x_exp, y_exp, w_exp, h_exp = rect_info["coordenadas_expandidas"]

        # Step 2: Crop ROI
        roi = image[y_exp : y_exp + h_exp, x_exp : x_exp + w_exp].copy()

        if roi.size == 0:
            print(f"ROI for ID {rect_info['id']} is empty. Skipping.")
            continue

        # Step 3: Apply FLD with RIDGE to the cropped ROI
        lines_roi, intersections_roi, rects_fld_roi, traps_fld_roi = (
            detector.process_roi(roi)
        )

        # Step 4: Map coordinates back to the original image
        # Map lines
        for line in lines_roi:
            all_fld_lines.append(
                [
                    int(line[0] + x_exp),
                    int(line[1] + y_exp),
                    int(line[2] + x_exp),
                    int(line[3] + y_exp),
                ]
            )

        # Map intersections
        for pt in intersections_roi:
            all_fld_intersections.append((int(pt[0] + x_exp), int(pt[1] + y_exp)))

        # Map shapes (rectangles and trapezoids)
        for r_pts in rects_fld_roi:
            mapped_r_pts = [(p[0] + x_exp, p[1] + y_exp) for p in r_pts]
            all_detected_fld_rects.append(mapped_r_pts)

        for t_pts in traps_fld_roi:
            mapped_t_pts = [(p[0] + x_exp, p[1] + y_exp) for p in t_pts]
            all_detected_fld_trapezoids.append(mapped_t_pts)

    # Apply NMS to the collected FLD shapes to remove duplicates/similar shapes
    final_fld_rects = detector.non_maximum_suppression_shapes(all_detected_fld_rects)
    final_fld_trapezoids = detector.non_maximum_suppression_shapes(
        all_detected_fld_trapezoids
    )

    # Step 5: Visualization
    output_fld_image = image.copy()

    # Draw all FLD lines
    for line in all_fld_lines:
        cv2.line(
            output_fld_image,
            (line[0], line[1]),
            (line[2], line[3]),
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )  # Yellow lines

    half = len(lines) // 2

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = map(int, line)

        cv2.line(black2_img, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(black2_img, (x1, y1), 3, (255, 255, 255), -1)
        cv2.circle(black2_img, (x2, y2), 3, (255, 255, 255), -1)

        cv2.line(black_img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.circle(black_img, (x1, y1), 3, (255, 255, 255), -1)
        # cv2.circle(black_img, (x2, y2), 3, (255, 255, 255), -1)

        """
        if i < half:
            xm = (x1 + x2) // 2
            ym = (y1 + y2) // 2

            # Dibujar la línea
            # Dibujar punto en el inicio
            cv2.circle(black2_img, (x1, y1), 3, (255, 255, 255), -1)
            # Dibujar punto en el fin
            cv2.circle(black2_img, (x2, y2), 3, (255, 255, 255), -1)
        """

        # Dibujar puntos solo en la mitad de las líneas
        if i % 2 == 0:  # por ejemplo en las líneas pares
            xm = (x1 + x2) // 2
            ym = (y1 + y2) // 2
            cv2.circle(black_img, (xm, ym), 3, (255, 255, 255), -1)

            # cv2.circle(black2_img, (xm, ym), 3, (255, 255, 255), -1)

    # Draw all FLD lines
    for line in all_fld_lines:
        cv2.line(
            black_img,
            (line[0], line[1]),
            (line[2], line[3]),
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw all FLD intersections
    for pt in all_fld_intersections:
        cv2.circle(output_fld_image, pt, 2, (0, 0, 255), -1)  # Red circles
        cv2.circle(black_img, pt, 2, (0, 0, 255), -1)  # Red circles

    # Draw final detected rectangles (white)
    #  for rect_pts in final_fld_rects:
    #      pts = np.array(rect_pts, np.int32).reshape((-1, 1, 2))
    #      cv2.polylines(output_fld_image, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw final detected trapezoids (light yellow)
    if detector.draw_trapezoids:
        for trap_pts in final_fld_trapezoids:
            pts = np.array(trap_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                output_fld_image, [pts], True, (255, 255, 255), 2, cv2.LINE_AA
            )  # Cyan

    return (
        output,
        output_filtrada,
        lines_img,
        rectangulos_info,
        black_img,
        output_fld_image,
        black2_img,
    )


def procesar_carpeta(input_folder, output_folder, expansion=10):
    """
    Procesa todas las imágenes de una carpeta

    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Crear subcarpetas para organizar resultados
    binary_folder = os.path.join(output_folder, "binarized")
    coords_folder = os.path.join(output_folder, "coordinates")

    if not os.path.exists(binary_folder):
        os.makedirs(binary_folder)
    if not os.path.exists(coords_folder):
        os.makedirs(coords_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath)

            if image is None:
                print(f"No se pudo leer {filename}")
                continue

            # Procesar imagen
            (
                resultado,
                resultado_filtrado,
                imagen_binarizada,
                rectangulos_info,
                grafica,
                roi,
                black,
            ) = detectar_rectangulos(image, expansion)

            # Guardar imagen con detecciones
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resultado)

            # Guardar imagen con detecciones
            output_path = os.path.join(output_folder, f"filtrada_{filename}")
            cv2.imwrite(output_path, resultado_filtrado)

            # Guardar imagen binarizada
            binary_path = os.path.join(binary_folder, f"binary_{filename}")
            cv2.imwrite(binary_path, imagen_binarizada)

            # Guardar imagen binarizada
            binary_path = os.path.join(binary_folder, f"lineas_{filename}")
            cv2.imwrite(binary_path, grafica)

            # Guardar imagen binarizada
            binary_path = os.path.join(binary_folder, f"black_{filename}")
            cv2.imwrite(binary_path, black)

            # Guardar imagen binarizada
            binary_path = os.path.join(binary_folder, f"roi_{filename}")
            cv2.imwrite(binary_path, roi)

            # Guardar coordenadas en archivo de texto
            base_name = os.path.splitext(filename)[0]
            coords_path = os.path.join(coords_folder, f"{base_name}_coordinates.txt")

            with open(coords_path, "w", encoding="utf-8") as f:
                f.write(f"Análisis de imagen: {filename}\n")
                f.write(f"Expansión del bounding box: {expansion} píxeles\n")
                f.write(f"Total de rectángulos detectados: {len(rectangulos_info)}\n\n")

                for rect in rectangulos_info:
                    f.write(f"--- Rectángulo ID: {rect['id']} ---\n")
                    f.write(f"Área: {rect['area']:.2f} píxeles²\n")

                    # Coordenadas originales
                    x, y, w, h = rect["coordenadas_originales"]
                    f.write(
                        f"Bounding Box Original: x={x}, y={y}, width={w}, height={h}\n"
                    )
                    f.write(f"Esquina superior izquierda: ({x}, {y})\n")
                    f.write(f"Esquina inferior derecha: ({x+w}, {y+h})\n")

                    # Coordenadas expandidas
                    x_exp, y_exp, w_exp, h_exp = rect["coordenadas_expandidas"]
                    f.write(
                        f"Bounding Box Expandido: x={x_exp}, y={y_exp}, width={w_exp}, height={h_exp}\n"
                    )
                    f.write(
                        f"Esquina superior izquierda expandida: ({x_exp}, {y_exp})\n"
                    )
                    f.write(
                        f"Esquina inferior derecha expandida: ({x_exp+w_exp}, {y_exp+h_exp})\n"
                    )

                    # Vértices del rectángulo (contorno original)
                    f.write(f"Vértices del contorno original:\n")
                    for j, vertex in enumerate(rect["vertices"]):
                        f.write(f"  Vértice {j+1}: ({vertex[0]}, {vertex[1]})\n")
                    f.write("\n")

            # Imprimir información en consola
            print(f"\n=== Procesando: {filename} ===")
            print(f"Rectángulos detectados: {len(rectangulos_info)}")

            for rect in rectangulos_info:
                x, y, w, h = rect["coordenadas_originales"]
                x_exp, y_exp, w_exp, h_exp = rect["coordenadas_expandidas"]

                print(f"\nRectángulo ID {rect['id']}:")
                print(f"  Coordenadas originales: ({x}, {y}, {w}, {h})")
                print(f"  Coordenadas expandidas: ({x_exp}, {y_exp}, {w_exp}, {h_exp})")
                print(f"  Área: {rect['area']:.2f} píxeles²")
                print(f"  Vértices del contorno:")
                for j, vertex in enumerate(rect["vertices"]):
                    print(f"    Vértice {j+1}: ({vertex[0]}, {vertex[1]})")

            print(f"\nArchivos guardados:")
            print(f"  - Imagen procesada: {output_path}")
            print(f"  - Imagen binarizada: {binary_path}")
            print(f"  - Coordenadas: {coords_path}")


def procesar_imagen_individual(image_path, expansion=10, mostrar_coordenadas=True):
    """
    Procesa una imagen individual y muestra el resultado

    Args:
        image_path: ruta de la imagen
        expansion: píxeles para expandir bounding box
        mostrar_coordenadas: si mostrar las coordenadas en los vértices
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return

    # Modificar la función para controlar si mostrar coordenadas
    resultado, edges, rectangulos_info = detectar_rectangulos(image, expansion)

    # Mostrar resultados
    # cv2.imshow("Imagen Original", image)
    # cv2.imshow("Detección de Rectángulos con Vértices", resultado)
    # cv2.imshow("Bordes Detectados", edges)

    print(f"\nRectángulos detectados: {len(rectangulos_info)}")
    for rect in rectangulos_info:
        print(f"\nRectángulo ID {rect['id']}:")
        print(f"  Vértices del contorno original:")
        for j, vertex in enumerate(rect["vertices"]):
            print(f"    Vértice {j+1}: ({vertex[0]}, {vertex[1]})")


#  cv2.waitKey(0)
#  cv2.destroyAllWindows()


# Ejemplo de uso
if __name__ == "__main__":

    input_folder = r"C:\Users\User\Desktop\Nueva carpeta MASTER\RED_TRAIN"
    output_folder = r"C:\Users\User\Desktop\Nueva carpeta MASTER\Wireframe-20250303T172646Z-001\Wireframe\resul110"

    # Puedes cambiar la expansión aquí (5, 10, 15, etc.)
    expansion_pixels = 10

    # Procesar toda la carpeta
    procesar_carpeta(input_folder, output_folder, expansion_pixels)

    # O procesar una imagen individual para pruebas
    # procesar_imagen_individual("ruta/a/tu/imagen.jpg", expansion_pixels)

    print("\n¡Procesamiento completado!")
    print(f"Revisa la carpeta '{output_folder}' para ver los resultados:")
    print(
        "- Las imágenes procesadas muestran círculos blancos en los vértices del contorno original"
    )
    print("- Los vértices están marcados con sus coordenadas")
