# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import cv2
import numpy as np

from application_util import preprocessing, visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

try:
    from tools.models import YOLODetector, ReIDModel
except ImportError as e:
    print(f"Failed to import models: {e}")
    print("Make sure 'torchreid' and 'ultralytics' are installed properly.")
    exit(1)

def gather_sequence_info(sequence_dir):
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {int(os.path.splitext(f)[0]): os.path.join(image_dir, f) for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())), cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = max_frame_idx = 0

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)
        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "update_ms": update_ms
    }
    return seq_info

def run(sequence_dir, detection_file, output_file, min_confidence, nms_max_overlap, min_detection_height, max_cosine_distance, nn_budget, display, yolo_model, reid_model, device):
    seq_info = gather_sequence_info(sequence_dir)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    yolo_detector = YOLODetector(model_type=yolo_model, device=device)
    reid_model = ReIDModel(model_name=reid_model, device=device)

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)
        image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

        detections = yolo_detector.detect(image)
        if len(detections) == 0:
            return

        boxes = detections[:, :4]
        confidences = detections[:, 4]

        features = reid_model.extract_features(image, boxes)
        detections = [Detection(bbox, conf, feat) for bbox, conf, feat in zip(boxes, confidences, features) if conf >= min_confidence]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        if display:
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    with open(output_file, 'w') as f:
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (row[0], row[1], row[2], row[3], row[4], row[5]), file=f)

def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid True/False choice")
    else:
        return input_string == "True"
def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument("--sequence_dir", help="Path to MOTChallenge sequence directory", default=None, required=True)
    parser.add_argument("--detection_file", help="Path to custom detections.", default=None)
    parser.add_argument("--output_file", help="Path to the tracking output file. This file will contain the tracking results on completion.", default="/tmp/hypotheses.txt")
    parser.add_argument("--min_confidence", help="Detection confidence threshold. Disregard all detections that have a confidence lower than this value.", default=0.8, type=float)
    parser.add_argument("--min_detection_height", help="Threshold on the detection bounding box height. Detections with height smaller than this value are disregarded", default=0, type=int)
    parser.add_argument("--nms_max_overlap", help="Non-maxima suppression threshold: Maximum detection overlap.", default=1.0, type=float)
    parser.add_argument("--max_cosine_distance", help="Gating threshold for cosine distance metric (object appearance).", type=float, default=0.2)
    parser.add_argument("--nn_budget", help="Maximum size of the appearance descriptors gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument("--display", help="Show intermediate tracking results", default=True, type=bool_string)
    parser.add_argument("--yolo_model", help="YOLO model type (yolov5s, yolov5m, yolov5l)", default="yolov5s.pt")
    parser.add_argument("--reid_model", help="ReID model name (osnet_x1_0, osnet_x0_75, osnet_x0_5)", default="osnet_x1_0")
    parser.add_argument("--device", help="Device to run models on (cpu, cuda)", default="cuda")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display,
        args.yolo_model, args.reid_model, args.device)
