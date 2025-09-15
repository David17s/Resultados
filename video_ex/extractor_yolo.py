import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt


import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


class RectangularObjectDetector:
    """Detector de objetos rectangulares usando YOLOv8 entrenado"""
    
    def __init__(self, model_path, confidence=0.25, iou_threshold=0.45):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = [
            'pintura', 'tv', 'caja', 'libro', 'brasero'
        ]
        self.colors = {
            'pintura': (255, 0, 0),      # Rojo
            'tv': (0, 255, 0),      # Verde
            'caja': (0, 0, 255),    # Azul
            'libro': (255, 255, 0), # Amarillo
            'brasero': (255, 0, 255),  # Magenta
        }
        
    def load_model(self):
        """Cargar modelo YOLOv8 entrenado"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Modelo cargado: {self.model_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
        return True
    
    def detect_objects(self, image_path, save_result=True, show_result=True):
        """Detectar objetos en una imagen"""
        if not self.model:
            if not self.load_model():
                return None
            
        
        # Determinar si es una ruta de archivo o una imagen numpy
        if isinstance(image_path, (str, Path)):
            # Es una ruta de archivo
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ No se pudo cargar la imagen: {iimage_path}")
                return None
            image_for_prediction = image_path
        else:
            # Es un array numpy (desde ROS2)
            image = image_path
            image_for_prediction = image
        
        # Cargar imagen
       # image = cv2.imread(str(image_path))
       # if image is None:
       #     print(f"❌ No se pudo cargar la imagen: {image_path}")
       #     return None
        
        # Hacer predicción
        results = self.model(
            image_path,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )


        # Áreas mínimas para cada clase personalizada
        min_area_by_class = {
            'pc': 3000,
            'tv': 4000,
            'caja': 500,
            'libro': 1000,
            'xray': 100
        }


        # Confianza mínima por clase
        min_confidence_by_class = {
            'pc': 0.5,
            'tv': 0.5,
            'caja': 0.5,
            'libro': 0.5,
            'xray': 0.5
        }
        
        # Procesar resultados
        detections = []
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extraer información
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                        color = self.colors.get(class_name, (0, 255, 255))
                        area = int((x2 - x1) * (y2 - y1))


                        # Verificar confianza mínima por clase
                        min_conf = min_confidence_by_class.get(class_name, 0)
                        if confidence < min_conf:
                            continue  # Saltar detección por baja confianza

                        # Verificar área mínima por clase
                        min_area = min_area_by_class.get(class_name, 0)
                        if area < min_area:
                            continue  # Saltar detección por ser muy pequeña
                        
                        # Guardar detección
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': [int((x1+x2)/2), int((y1+y2)/2)],
                            'area': area
                        }
                        detections.append(detection)
                        
                        # Dibujar bounding box
                        cv2.rectangle(annotated_image, (int(x1)-15, int(y1)-15), (int(x2)+15, int(y2)+15), color, 2)
                        
                        # Dibujar etiqueta
                        label = f"{class_name}: {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(
                            annotated_image, 
                            (int(x1)-15, int(y1)-15 - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)-15),
                            color, -1
                        )
                        cv2.putText(
                            annotated_image, label,
                            (int(x1)-15, int(y1) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                        )

                        # Obtener dimensiones de la imagen
                        h, w = image.shape[:2]

                        # Calcular coordenadas seguras para la ROI con padding
                        padding = 15
                        x1_safe = max(0, int(x1) - padding)
                        y1_safe = max(0, int(y1) - padding)
                        x2_safe = min(w, int(x2) + padding)
                        y2_safe = min(h, int(y2) + padding)

                        # Verificar que la ROI tenga dimensiones válidas
                        if x2_safe > x1_safe and y2_safe > y1_safe:
                            # Extraer ROI
                            roi = image[y1_safe:y2_safe, x1_safe:x2_safe]
                            
                            # Verificar que la ROI no esté vacía
                            if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                                try:
                                    # Convertir a escala de grises
                                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                    gray = cv2.convertScaleAbs(gray, alpha=1, beta=0)  # Aumentar contraste

                                    # Detectar bordes
                                    edges = cv2.Canny(gray, 30, 100)
                                    
                                    # Morfología para cerrar contornos
                                    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                                    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)

                                    # Encontrar contornos
                                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    
                                    if contours:
                                        # Obtener el contorno más grande
                                        largest = max(contours, key=cv2.contourArea)
                                        epsilon = 0.04 * cv2.arcLength(largest, True)
                                        approx = cv2.approxPolyDP(largest, epsilon, True)

                                        # Si encontramos un cuadrilátero (4 vértices)
                                        if len(approx) == 4:
                                            # Convertir coordenadas locales (ROI) a coordenadas globales (imagen completa)
                                            pts = approx.reshape(-1, 2) + np.array([x1_safe, y1_safe])
                                            vertices = pts

                                            # Dibujar líneas blancas entre cada par de puntos consecutivos
                                            for i in range(len(vertices)):
                                                pt1 = tuple(vertices[i].astype(int))
                                                pt2 = tuple(vertices[(i + 1) % len(vertices)].astype(int))  # el % cierra el polígono
                                                cv2.line(annotated_image, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

                                            # Dibujar los vértices
                                            for i, vertex in enumerate(vertices):
                                                punto = tuple(vertex.astype(int))
                                                # Dibujar el vértice (círculo blanco con borde negro)
                                                cv2.circle(annotated_image, punto, 5, (255, 255, 255), -1)
                                                cv2.circle(annotated_image, punto, 6, (0, 0, 0), 2)
                                                
                                            # Opcional: agregar información de los vértices a la detección
                                          #  detection['vertices'] = vertices.tolist()
                                            
                                except Exception as e:
                                    print(f"⚠️ Error procesando ROI para {class_name}: {e}")
                            else:
                                print(f"⚠️ ROI vacía para detección {class_name}")
                        else:
                            print(f"⚠️ Coordenadas inválidas para detección {class_name} - bbox: ({x1}, {y1}, {x2}, {y2})")

                        '''
                        

                        # ---------------------
                        # 🔍 Detección de contornos en la ROI
                        roi = image[int(y1-):int(y2+15), int(x1-15):int(x2+15)]
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                        gray = cv2.convertScaleAbs(gray, alpha=1, beta=0) # Aumentar contraste

                        # 2. Aplicar un filtro bilateral para preservar los bordes mientras se suaviza el ruido
                        # Esto es mejor que GaussianBlur para bordes
                      #  blurred = cv2.bilateralFilter(gray, 9, 75, 75)
                        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                       


                        # set the kernel size, depending on whether we are using the Sobel
                        # filter or the Scharr operator, then compute the gradients along
                        edges = cv2.Canny(gray, 30, 100)
                       # gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
                        #gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
                        # the gradient magnitude images are now of the floating point data
                        # type, so we need to take care to convert them back a to unsigned
                        # 8-bit integer representation so other OpenCV functions can operate
                        # on them and visualize them
                        #gX = cv2.convertScaleAbs(gX)
                        #gY = cv2.convertScaleAbs(gY)
                        # combine the gradient representations into a single image
                        #edges = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
                     
                       
                        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)

                        #kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        #edges= cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_connect, iterations=1)

                        # 5. Dilatar ligeramente para asegurar conectividad
                       # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                       # edges = cv2.dilate(edges, kernel_dilate, iterations=1)

                      #  self.show_result(annotated_image, edges, detections, Path(image_path).name)

                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            epsilon = 0.04 * cv2.arcLength(largest, True)
                            approx = cv2.approxPolyDP(largest, epsilon, True)

                            if len(approx) == 4:
                                pts = approx.reshape(-1, 2) + np.array([int(x1)-15, int(y1)-15])

                                vertices = pts

                                  # Dibujar líneas blancas entre cada par de puntos consecutivos
                                for i in range(len(vertices)):
                                    pt1 = tuple(vertices[i].astype(int))
                                    pt2 = tuple(
                                        vertices[(i + 1) % len(vertices)].astype(int)
                                    )  # el % cierra el polígono
                                    cv2.line(annotated_image, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)


                                for i, vertex in enumerate(vertices):
                                    punto = tuple(vertex.astype(int))

                                    # Dibujar el vértice
                                    cv2.circle(annotated_image, punto, 5, (255, 255, 255), -1)
                                    cv2.circle(annotated_image, punto, 6, (0, 0, 0), 2)

                              
                                #for i in range(4):
                                #    pt1 = tuple(pts[i])
                                #    pt2 = tuple(pts[(i + 1) % 4])
                                #    cv2.line(annotated_image, pt1, pt2, (255, 255, 255), 2)
                                #    cv2.circle(annotated_image, pt1, 5, (255, 255, 255), -1)
                        '''
        
        # Guardar resultado
        #if save_result:
        #    output_path = Path(image_path).parent / f"detected_{Path(image_path).name}"
        #    cv2.imwrite(str(output_path), annotated_image)
        #    print(f"💾 Resultado guardado en: {output_path}")
        
        # Mostrar resultado
        #if show_result:
        #    self.show_result(image, annotated_image, detections, Path(image_path).name)
            
        
        return {
            'detections': detections,
            'annotated_image': annotated_image,
            'original_image': image
        }
    
    def show_result(self, original, annotated, detections, title):
        """Mostrar resultado de detección"""
        plt.figure(figsize=(15, 8))
        
        # Imagen original
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title(f'Original - {title}')
        plt.axis('off')
        
        # Imagen con detecciones
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f'Detecciones: {len(detections)} objetos')
        plt.axis('off')
        
        # Información de detecciones
        if detections:
            info_text = []
            class_counts = {}
            for det in detections:
                class_name = det['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in class_counts.items():
                info_text.append(f"{class_name}: {count}")
            
            plt.figtext(0.5, 0.02, " | ".join(info_text), ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def detect_in_directory(self, input_dir, output_dir=None, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """Detectar objetos en todas las imágenes de un directorio"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Directorio no existe: {input_dir}")
            return
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path / "detections"
            output_path.mkdir(exist_ok=True)
        
        # Buscar imágenes
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No se encontraron imágenes en: {input_dir}")
            return
        
        print(f"🔍 Procesando {len(image_files)} imágenes...")
        
        all_detections = []
        for img_file in image_files:
            print(f"Procesando: {img_file.name}")
            result = self.detect_objects(img_file, save_result=False, show_result=False)
            
            if result:
                # Guardar imagen anotada
                output_file = output_path / f"detected_{img_file.name}"
                #cv2.imwrite(str(output_file), result['annotated_image'])
                
                # Recopilar estadísticas
                for detection in result['detections']:
                    detection['image'] = img_file.name
                    all_detections.append(detection)
        
        # Mostrar estadísticas
        self.show_batch_statistics(all_detections)
        
        print(f"✅ Procesamiento completado. Resultados en: {output_path}")
    
    def show_batch_statistics(self, all_detections):
        """Mostrar estadísticas del procesamiento en lote"""
        print("\n📊 ESTADÍSTICAS DE DETECCIÓN")
        print("="*40)
        
        if not all_detections:
            print("No se encontraron detecciones")
            return
        
        # Contar por clase
        class_counts = {}
        confidence_by_class = {}
        
        for det in all_detections:
            class_name = det['class']
            confidence = det['confidence']
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if class_name not in confidence_by_class:
                confidence_by_class[class_name] = []
            confidence_by_class[class_name].append(confidence)
        
        print(f"Total detecciones: {len(all_detections)}")
        print("\nDetecciones por clase:")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            confidences = confidence_by_class[class_name]
            avg_conf = np.mean(confidences)
            
            print(f"  {class_name:<8}: {count:>3} detecciones (conf. promedio: {avg_conf:.3f})")
        
        # Distribución de confianza
        all_confidences = [det['confidence'] for det in all_detections]
        print(f"\nConfianza general:")
        print(f"  Promedio: {np.mean(all_confidences):.3f}")
        print(f"  Mediana:  {np.median(all_confidences):.3f}")
        print(f"  Min/Max:  {np.min(all_confidences):.3f}/{np.max(all_confidences):.3f}")



class VideoExtractorNode(Node):
    def __init__(self):
        super().__init__('video_extractor_node')

        self.subscription = self.create_subscription(Image,'/rgb',self.listener_callback,10)
        
        self.bridge = CvBridge()

          # Crear detector
        self.detector = RectangularObjectDetector(
            model_path="/home/zzh/isaacsim/standalone_examples/replicator/custom_scene/rectangular_objects2/train3/weights/best.pt",
            confidence=0.5,
            iou_threshold=0.3
        )

        
        #self.out = cv2.VideoWriter('rgb_depth_output.avi',
        #                           cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))
        

    
        self.get_logger().info("Inicio video. Escuchando /rgb_depth")

    
    def listener_callback(self,msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Redimensionar el frame a 512x512
            resized_frame = cv2.resize(frame, (512, 512))

        
            # Obtener resultados de detección
            result = self.detector.detect_objects(resized_frame, save_result=False, show_result=False)
            
            if result is not None:
                detections = result['detections']
                annotated_image = result['annotated_image']
                original_image = result['original_image']
                
                cv2.imshow("RGB", original_image)
                cv2.imshow("DETECCIONES", annotated_image)
            else:
                # Si falla la detección, mostrar imagen original
                cv2.imshow("RGB", resized_frame)
           
            key = cv2.waitKey(1)


            # Si se presiona la tecla '1' (código ASCII 49)
            if key == 49:  # 49 es el código ASCII para la tecla '1'
                # Definir la ruta de la carpeta de capturas
                save_folder = '/home/zzh/isaacsim/capturas_ros2'
                
                # Crear la carpeta si no existe
                import os
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                    print(f"Carpeta creada: {save_folder}")
                
                # Generar un nombre de archivo único basado en la fecha/hora
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Incluimos milisegundos
                filename = os.path.join(save_folder, f"capture_{timestamp}.png")
                
                # Guardar la imagen
                cv2.imwrite(filename, resized_frame)
                print(f"Imagen guardada como {filename}")


        except Exception as e:
            self.get_logger().error(f"Error: {e}")

     

    def destroy_node(self):
        self.out.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoExtractorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


