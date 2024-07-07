# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import torch
from models import YOLODetector, ReIDModel

def extract_image_patch(image, bbox, patch_shape):
    bbox = np.array(bbox)
    if patch_shape is not None:
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int64)
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image

def generate_detections(yolo_detector, reid_model, mot_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)
        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {int(os.path.splitext(f)[0]): os.path.join(image_dir, f) for f in os.listdir(image_dir)}

        detections_out = []

        for frame_idx in sorted(image_filenames.keys()):
            print("Frame %05d" % frame_idx)
            image_path = image_filenames[frame_idx]
            image = cv2.imread(image_path)

            detections = yolo_detector.detect(image)
            if len(detections) == 0:
                continue

            boxes = detections[:, :4]
            confidences = detections[:, 4]

            features = reid_model.extract_features(image, boxes)
            for box, confidence, feature in zip(boxes, confidences, features):
                detections_out.append(np.r_[frame_idx, box, confidence, feature.cpu().numpy()])

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(output_filename, np.asarray(detections_out), allow_pickle=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument("--yolo_model", help="YOLO model type (yolov5s, yolov5m, yolov5l)", default="yolov5s")
    parser.add_argument("--reid_model", help="ReID model name (osnet_x1_0, osnet_x0_75, osnet_x0_5)", default="osnet_x1_0")
    parser.add_argument("--device", help="Device to run models on (cpu, cuda)", default="cpu")
    parser.add_argument("--mot_dir", help="Path to MOTChallenge directory (train or test)", required=True)
    parser.add_argument("--output_dir", help="Output directory. Will be created if it does not exist.", default="detections")
    return parser.parse_args()

def main():
    args = parse_args()
    yolo_detector = YOLODetector(model_type=args.yolo_model, device=args.device)
    reid_model = ReIDModel(model_name=args.reid_model, device=args.device)
    generate_detections(yolo_detector, reid_model, args.mot_dir, args.output_dir)

if __name__ == "__main__":
    main()
