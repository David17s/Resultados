import json
from shapely.geometry import Polygon
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# === Parámetros ===
detections_file = r"C:\\Users\\User\\Desktop\\Nueva carpeta MASTER\\SessionTotal1\\detections_combined_rectangles.json"
annotation_files = [
    r"C:\Users\User\Desktop\Nueva carpeta MASTER\RED_TRAIN\valid\_annotations.coco.json",
    r"C:\Users\User\Desktop\Nueva carpeta MASTER\RED_TRAIN\test\_annotations.coco.json",
    r"C:\Users\User\Desktop\Nueva carpeta MASTER\RED_TRAIN\train\_annotations.coco.json"
]
image_folder = r"C:\Users\User\Desktop\Nueva carpeta MASTER\RED_Test"
output_folder = r"C:\Users\User\Desktop\Nueva carpeta MASTER\visualizations"
os.makedirs(output_folder, exist_ok=True)

# Parámetros de evaluación
iou_threshold = 0.5
min_area_ratio = 0.0005
min_iou_for_fp = 0.1  # Umbral para considerar FP

def non_max_suppression(polygons, iou_thresh=0.5):
    polys = sorted(polygons, key=lambda p: p.area, reverse=True)
    keep = []
    for poly in polys:
        should_keep = True
        for kept in keep:
            inter = poly.intersection(kept).area
            union = poly.union(kept).area
            if union > 0 and inter / union > iou_thresh:
                should_keep = False
                break
        if should_keep:
            keep.append(poly)
    return keep

def visualize_results(image_path, preds, gts, matched_gt_indices, output_path):
    """Visualiza TP, FP y FN en la imagen y guarda el resultado"""
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)
    
    # Dibujar ground truth (GT)
    for i, gt in enumerate(gts):
        if i in matched_gt_indices:
            continue  # Los TP se dibujan con las predicciones
        x, y = gt.exterior.xy
        ax.add_patch(patches.Polygon(xy=list(zip(x, y)), 
                     fill=False, 
                     edgecolor='yellow', 
                     linewidth=2, 
                     linestyle='--', 
                     label='FN' if i == 0 else ''))
    
    # Dibujar predicciones
    tp_patches = 0
    fp_patches = 0
    for pred in preds:
        is_tp = False
        for i, gt in enumerate(gts):
            if i not in matched_gt_indices:
                continue
            iou = pred.intersection(gt).area / pred.union(gt).area
            if iou >= iou_threshold:
                is_tp = True
                break
        
        x, y = pred.exterior.xy
        if is_tp:
            ax.add_patch(patches.Polygon(xy=list(zip(x, y)), 
                         fill=False, 
                         edgecolor='blue', 
                         linewidth=2, 
                         label='TP' if tp_patches == 0 else ''))
            tp_patches += 1
        else:
            # Verificar si es FP válido (IoU > min_iou_for_fp)
            is_valid_fp = any((pred.intersection(gt).area / pred.union(gt).area > min_iou_for_fp 
                             for gt in gts))
            if is_valid_fp:
                ax.add_patch(patches.Polygon(xy=list(zip(x, y)), 
                             fill=False, 
                             edgecolor='red', 
                             linewidth=2, 
                             label='FP' if fp_patches == 0 else ''))
                fp_patches += 1
    for i, gt in enumerate(gts):
        x, y = gt.exterior.xy
        if i in matched_gt_indices:
            ax.add_patch(patches.Polygon(xy=list(zip(x, y)), fill=False, edgecolor='green', linewidth=2, label='GT (TP)' if i==0 else ''))
        else:
            ax.add_patch(patches.Polygon(xy=list(zip(x, y)), fill=False, edgecolor='yellow', linewidth=2, linestyle='--', label='GT (FN)' if i==0 else ''))

    # Configurar leyenda
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.title(os.path.basename(image_path))
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

# --- Cargar detecciones ---
detections = json.load(open(detections_file))
detections = {os.path.basename(k): v for k, v in detections.items()}
print(f"Total detecciones cargadas: {len(detections)}")

# --- Cargar y unir anotaciones COCO ---
merged = {"images": [], "annotations": [], "categories": []}
img_off = ann_off = 0
for filepath in annotation_files:
    data = json.load(open(filepath))
    for img in data['images']:
        img['id'] += img_off
        merged['images'].append(img)
    for ann in data['annotations']:
        ann['image_id'] += img_off
        ann['id'] += ann_off
        merged['annotations'].append(ann)
    if not merged['categories'] and 'categories' in data:
        merged['categories'] = data['categories']
    img_off = max(i['id'] for i in merged['images']) + 1
    ann_off = max(a['id'] for a in merged['annotations']) + 1

# Obtener nombres reales desde "extra"
id2fn = {img['id']: img['extra']['name'] for img in merged['images'] if 'extra' in img}

# Agrupar GT por imagen
gt_by_image = {}
for ann in merged['annotations']:
    fn = id2fn.get(ann['image_id'])
    if not fn:
        continue
    if isinstance(ann['segmentation'], list) and ann['segmentation']:
        seg = ann['segmentation'][0]
        if isinstance(seg, list) and len(seg) >= 6:
            pts = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            poly = Polygon(pts)
            if poly.is_valid:
                gt_by_image.setdefault(fn, []).append(poly)

# Filtrar imágenes que tienen anotaciones y detecciones
imagenes_a_evaluar = set(gt_by_image.keys()) & set(detections.keys())

tp = fp = fn = 0

for filename in imagenes_a_evaluar:
    gts = gt_by_image[filename]
    preds = []
    for r in detections.get(filename, []):
        poly = Polygon(r)
        if poly.is_valid and poly.area >= min_area_ratio:
            preds.append(poly)
    preds = non_max_suppression(preds, iou_thresh=0.5)
    matched = set()
    
    # Primera pasada: encontrar TP
    for pred in preds:
        for i, gt in enumerate(gts):
            if i in matched:
                continue
            if not pred.is_valid or not gt.is_valid:
                continue
            iou = pred.intersection(gt).area / pred.union(gt).area
            if iou >= iou_threshold:
                tp += 1
                matched.add(i)
                break
    
    # Segunda pasada: contar FP válidos
    for pred in preds:
        is_fp = True
        # Si ya es TP, no es FP
        for i, gt in enumerate(gts):
            if i in matched:
                iou = pred.intersection(gt).area / pred.union(gt).area
                if iou >= iou_threshold:
                    is_fp = False
                    break
        # Verificar si es FP válido
        if is_fp and any((pred.intersection(gt).area / pred.union(gt).area > min_iou_for_fp for gt in gts)):
            fp += 1
    
    fn += len(gts) - len(matched)
    
    # Visualizar resultados para esta imagen
    img_path = os.path.join(image_folder, filename)
    output_path = os.path.join(output_folder, f"result_{filename}")
    visualize_results(img_path, preds, gts, matched, output_path)

# Resultados
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📊 Evaluación solo en imágenes anotadas con detecciones:")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
total_gt = sum(len(gt_by_image[fn]) for fn in imagenes_a_evaluar)
print(f"Total de anotaciones reales (GT) en imágenes evaluadas: {total_gt}")

