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

            result_frame = self.detect_shapes(frame.copy())
          #  self.out.write(result_frame)
           
            cv2.imshow("RGB Depth", frame)
            cv2.imshow("Shape Detection", result_frame)
           
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    
    def detect_shapes(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

        cv2.imshow("mascara", thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                vertices = len(approx)

                x, y, w, h = cv2.boundingRect(approx)

                shape_type = "Unidentified"
                if vertices == 3:
                    shape_type = "Triangle"
                elif vertices == 4:
                    aspect_ratio = float(w) / h
                    if 0.95 <= aspect_ratio <= 1.05:
                        shape_type = "Square"
                    else:
                        shape_type = "Rectangle"
                elif vertices > 4:
                    shape_type = "Circle"

                # Dibuja contorno y etiqueta
                cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
                cv2.putText(img, shape_type, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return img
        

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


