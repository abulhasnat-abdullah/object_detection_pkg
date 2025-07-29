import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import shutil
from datetime import datetime

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class Detectron2ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('detectron2_object_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera_front/image_raw',  # Update if topic changes
            self.listener_callback,
            10
        )

        self.bridge = CvBridge()
        self.predictor = self.load_model()

        self.image_dir = "detected_images"
        self.csv_path = "detection_log.csv"
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir, exist_ok=True)

        with open(self.csv_path, mode='w') as f:
            f.write(f"{'Timestamp'.ljust(24)}, {'Object Class'.ljust(15)}, Image Path\n")

        self.get_logger().info("Detectron2ObjectDetectionNode initialized and subscribed to /camera_front/image_raw/ffmpeg")

    def load_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return DefaultPredictor(cfg)

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        outputs = self.predictor(frame)
        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().tolist()
        pred_boxes = instances.pred_boxes.tensor.cpu().tolist()
        scores = instances.scores.cpu().tolist()

        if not pred_classes:
            return

        annotations = []
        metadata = MetadataCatalog.get(self.predictor.cfg.DATASETS.TRAIN[0])
        for cls_id, box, score in zip(pred_classes, pred_boxes, scores):
            label = metadata.get("thing_classes", [])[cls_id]
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_label = y1 - th - 10 if (y1 - th - 10) > 10 else y1 + th + 10
            cv2.rectangle(frame, (x1, y1_label - th - 6), (x1 + tw + 4, y1_label), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1_label - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            annotations.append(label)

        img_filename = f"{self.image_dir}/{int(datetime.now().timestamp() * 1000)}.jpg"
        cv2.imwrite(img_filename, frame)

        with open(self.csv_path, mode='a') as f:
            for label in annotations:
                f.write(f"{timestamp.ljust(24)}, {label.ljust(15)}, {img_filename}\n")


def main(args=None):
    rclpy.init(args=args)
    node = Detectron2ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
