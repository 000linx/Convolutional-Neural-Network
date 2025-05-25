from ultralytics import YOLO
from config import config
import cv2


def detect_face(image_path):
    detector = YOLO(config.DETECTOR_MODEL_PATH)
    results = detector(image_path)
    boxes = results[0].boxes.xyxy.cuda().numpy()
    return boxes