import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import os
import shutil
from datetime import datetime, timedelta

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera_front/image_raw',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.model = YOLO("yolov8x.pt")
        self.device = "cuda" if self.model.device.type == "cuda" else "cpu"

        self.image_dir = "detected_images"
        self.csv_path = "detection_log.csv"
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir, exist_ok=True)

        with open(self.csv_path, mode='w') as f:
            f.write(f"{'Timestamp'.ljust(24)}, {'Object Class'.ljust(15)}, Image Path\n")

        self.last_seen = {}
        self.cooldown = timedelta(seconds=5)

        self.get_logger().info("ObjectDetectionNode initialized and listening to /camera_front/image_raw/ffmpeg")

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        results = self.model(frame, verbose=False)[0]
        if results.boxes is None or len(results.boxes.cls) == 0:
            return

        save_needed = False
        labels_to_save = []

        for box in results.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            label = self.model.names[cls_id]
            last_time = self.last_seen.get(label)

            if last_time and now - last_time < self.cooldown:
                continue  # skip both saving and logging

            self.last_seen[label] = now
            labels_to_save.append((label, box))
            save_needed = True

        if not save_needed:
            return  # no new detections to save

        # Draw boxes and save image
        for label, box in labels_to_save:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_label = y1 - th - 10 if (y1 - th - 10) > 10 else y1 + th + 10

            cv2.rectangle(frame, (x1, y1_label - th - 6), (x1 + tw + 4, y1_label), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1_label - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        img_filename = f"{self.image_dir}/{int(now.timestamp() * 1000)}.jpg"
        cv2.imwrite(img_filename, frame)

        with open(self.csv_path, mode='a') as f:
            for label, _ in labels_to_save:
                f.write(f"{timestamp_str.ljust(24)}, {label.ljust(15)}, {img_filename}\n")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
