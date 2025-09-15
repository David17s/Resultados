
import os
import cv2
import numpy as np
import torch
import math
from glob import glob
from itertools import combinations
from video_extractor.unet import UNet
from video_extractor.otros.deximodel import DexiNed

# === Configuración ===
USING_UNET = False
DRAW_TRAPEZOIDS = True

# === Paths de modelo ===
DEXINED_PATH = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/dexi.pth'
UNET_PATH    = '/home/zzh/ros2_ws/src/video_extractor/video_extractor/checkpoints/unet.pth'

# === Parámetros de detección ===
THETA_RESOLUTION = 6
KERNEL_SIZE = 9
ELONGATION_FACTOR = 1.5
RECTS_MIN_AREA_FACTOR = 0.001
RECTS_MAX_AREA_FACTOR = 0.1
COMPACTNESS_THRESHOLD = 0.4
NMS_THRESHOLD = 10
WHITENESS_THRESHOLD = 0.7
MAX_NUM_TRAPEZOIDS = 50
RECT_DRAWING_WIDTH = 2

def init_orientation_detector(theta_resolution, kernel_size, device):
    od = torch.nn.Conv2d(1, theta_resolution, kernel_size, stride=1, padding=kernel_size//2, bias=False).to(device)
    for i in range(theta_resolution):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        angle = i * 180 / theta_resolution
        dx = int(np.cos(np.deg2rad(angle)) * (kernel_size // 2))
        dy = int(np.sin(np.deg2rad(angle)) * (kernel_size // 2))
        cv2.line(kernel, (kernel_size//2 - dx, kernel_size//2 - dy), (kernel_size//2 + dx, kernel_size//2 + dy), 1, 1)
        od.weight.data[i, 0] = torch.from_numpy(kernel)
    return od

class ImageProcessor:
    def __init__(self, model, orientation_detector, device):
        self.model = model
        self.orientation_detector = orientation_detector
        self.device = device

    # resto del código ...


    def process(self, image):
        self.image = image.copy()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = self.image.shape[:2]
        new_h, new_w = (h // 16) * 16, (w // 16) * 16
        image = cv2.resize(self.image, (new_w, new_h))

        inp = image if not USING_UNET else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inp = torch.from_numpy(inp).to(self.device).float() / 255.0
        if USING_UNET:
            inp = inp.unsqueeze(0).unsqueeze(0)
        else:
            inp = inp.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(inp)
            orient = self.orientation_detector(pred)
            _ = torch.nn.functional.normalize(orient - orient.mean(1, keepdim=True), p=2.0, dim=1)

        edge_map = pred.detach().cpu().numpy()[0, 0]
        thresh = 0.5 if not USING_UNET else -2.5
        edges = (edge_map > thresh).astype(np.uint8) * 255

        self.edges = edges
        self.lines = self.detect_lines()

        # Imagen para líneas
        self.lines_img = self.image.copy()
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(self.lines_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # Imagen para rectángulos
        self.rect_img = self.image.copy()
        self.elongate_lines()
        self.find_intersections()
        self.create_graph()
        self.extract_trapezoids()
        self.draw_trapezoids_func()

        return {
            'original': self.image,
            'lines_img': self.lines_img,
            'rectangles': self.rect_img,
        }


def main(folder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(1, 1).to(device) if USING_UNET else DexiNed().to(device)
    model.load_state_dict(torch.load(UNET_PATH if USING_UNET else DEXINED_PATH, map_location=device))
    model.eval()
    orientation_detector = init_orientation_detector(THETA_RESOLUTION, KERNEL_SIZE, device)
    processor = ImageProcessor(model, orientation_detector, device)

    output_folder = os.path.join(folder, "output")
    os.makedirs(output_folder, exist_ok=True)

    image_paths = sorted(glob(os.path.join(folder, '*.*')))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Error al cargar {path}")
            continue

        outputs = processor.process(img)

        base_name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_original.png"), outputs['original'])
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_lines.png"), outputs['lines_img'])
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_rectangles.png"), outputs['rectangles'])

        cv2.imshow("Original", outputs['original'])
        cv2.imshow("Lines", outputs['lines_img'])
        cv2.imshow("Rectangles", outputs['rectangles'])

        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    folder = "/home/zzh/rh2/rgbd/session_1/alma/fullhouse1"
    main(folder)
