import json 
import os

def coco_to_hawrp_per_image(coco_json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    
    # Crear un diccionario para agrupar anotaciones por imagen
    image_annotations = {}
    
    # Mapear image_id a file_name para usar después
    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if 'segmentation' not in ann or not ann['segmentation']:
            continue
        
        polygons_hawrp = []
        for polygon in ann['segmentation']:
            coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            junctions = coords
            fields = [(k, (k+1) % len(coords)) for k in range(len(coords))]
            polygons_hawrp.append({
                "junctions": junctions,
                "fields": fields
            })
        
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].extend(polygons_hawrp)
    
    # Guardar un archivo JSON por imagen con "filename"
    for img_id, polygons in image_annotations.items():
        file_name = id_to_filename[img_id]
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, base_name + "_hawrp.json")
        
        output_data = {
            "filename": file_name,
            "polygons": polygons
        }
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(output_data, f_out, indent=2)
    
    print(f"✅ Archivos HAWRP guardados en {output_dir}")

# ==== Configura tus rutas ====
coco_path = "C:\\Users\\User\\Desktop\\Nueva carpeta MASTER\\Rectangulos OBB.v3i.coco-mmdetection\\valid\\_annotations.coco.json"
output_folder = "C:\\Users\\User\\Desktop\\Nueva carpeta MASTER\\Rectangulos OBB.v3i.coco-mmdetection\\hawrp_valid"

coco_to_hawrp_per_image(coco_path, output_folder)
