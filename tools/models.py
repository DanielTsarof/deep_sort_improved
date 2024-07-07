import torch
from torchvision import transforms
from torchreid.utils import FeatureExtractor
from ultralytics import YOLO
import numpy as np
from matplotlib import cm
from PIL import Image

class YOLODetector:
    def __init__(self, model_type='yolov5s', device='cpu'):
        self.device = device
        self.model = YOLO(model_type).to(device)

    def detect(self, image):
        results = self.model(image)
        boxes = results[0].boxes
        detections = []
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            cls = box.cls[0].item()
            detections.append([x1, y1, x2, y2, confidence, cls])
        return np.array(detections)


class ReIDModel:
    def __init__(self, model_name='osnet_x1_0', device='cpu'):
        self.device = device
        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=device
        )

    def extract_features(self, image, boxes):
        image = Image.fromarray(image)
        crops = [transforms.functional.crop(image, int(box[1]), int(box[0]), int(box[3] - box[1]), int(box[2] - box[0]))
                 for box in boxes]
        print("******", crops, type(crops))
        crops = np.array(crops)
        features = self.extractor(crops)
        return features