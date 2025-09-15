import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np



class VideoExtractorNode(Node):
    def __init__(self):
        super().__init__('video_extractor_node')

        self.subscription = self.create_subscription(Image,'/rgb',
self.listener_callback,10)
        
        self.bridge = CvBridge()

        
        #self.out = cv2.VideoWriter('rgb_depth_output.avi',
        #                           cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))
        

    
        self.get_logger().info("Inicio video. Escuchando /rgb_depth")

    
    def listener_callback(self,msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Redimensionar el frame a 512x512
            resized_frame = cv2.resize(frame, (512, 512))

            
            cv2.imshow("RGB", frame)
           
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


