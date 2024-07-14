from __future__ import division, print_function, absolute_import

import argparse
import os
import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from PIL import Image

from nn_models.yolov5_detect import load_yolo_model, detect_objects, xywh_to_ltwh
from nn_models.reid_model import load_reid_model
from nn_models.segmentation import Segmentation


def gather_sequence_info(sequence_dir, detection_file=None):
    """Gather sequence information, such as image filenames and groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image filenames.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.
        * detection_file: path to file with detection info of None

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())

    if detections is not None:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections_yolo(yolo_model, reid_model, img, min_confidence):
    """Create detections for a given image using YOLO and ReID model.

    Parameters
    ----------
    yolo_model : YOLOv5 model
        The YOLOv5 model for detection.
    reid_model : ReID model
        The ReID model for feature extraction.
    img : numpy.ndarray
        Image array.
    min_confidence : float
        Minimum confidence threshold for detections.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses for the given image.
    """
    results = detect_objects(yolo_model, img)
    detections = []

    for *xywh, conf, cls in results.xywh[0].cpu().numpy():
        if int(cls) == 0 and conf >= min_confidence:  # Filter for 'person' class
            ltwh = xywh_to_ltwh(xywh)
            if ltwh[2] > 0 and ltwh[3] > 0:  # Ensure valid bounding box
                x1, y1, w, h = ltwh
                bbox_img = img[y1:y1 + h, x1:x1 + w]
                if bbox_img.size > 0:
                    bbox_pil = Image.fromarray(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
                    features = reid_model.extract_features(bbox_pil).numpy().flatten()
                    detections.append(Detection(ltwh, conf, features))
    return detections


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int64)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, detection_model, reid_model_type, segmentation):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.
    detection_model : str
        Type of used detection model.
    reid_model_type : str
        Type of used ReID model.
    segmentation : bool
        Flag to apply segmentation to the video.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    if segmentation:
        segmenter = Segmentation()
        mask_save_dir = os.path.join(output_file, "masks")
        os.makedirs(mask_save_dir, exist_ok=True)

        def segment_only_callback(vis, frame_idx):
            print("Processing frame %05d for segmentation only" % frame_idx)

            # Load image
            image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

            # Apply segmentation
            mask = segmenter.segment_person(image)
            save_path = os.path.join(mask_save_dir, f"mask_{frame_idx:05d}.png")
            image = segmenter.overlay_mask(image, mask, (0, 0, 255))  # Red color for mask
            cv2.imwrite(save_path, mask * 255)
            if display:
                vis.set_image(image.copy())

        visualizer = visualization.Visualization(seq_info, update_ms=5) if display else visualization.NoVisualization(
            seq_info)
        visualizer.run(segment_only_callback)
        return

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    yolo_model = None
    reid_model = None

    if detection_file is None:
        yolo_model = load_yolo_model(f"nn_models/weights/yolo/{detection_model}.pt")
        reid_model = load_reid_model(model_type=reid_model_type)

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        if detection_file is None:
            detections = create_detections_yolo(yolo_model, reid_model, image, min_confidence)
        else:
            detections = create_detections(
                seq_info["detections"], frame_idx, min_detection_height)
            detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    with open(os.path.join(output_file, f'{sequence_dir.split(os.sep)[-1]}.txt'), 'w') as f:
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]), file=f)


def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid True/False choice")
    else:
        return input_string == "True"


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
                              " contain the tracking results on completion.",
        default="./tmp")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
                                 "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
                                       "box height. Detections with height smaller than this value are "
                                       "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap", help="Non-maxima suppression threshold: Maximum "
                                  "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
                                      "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
                            "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--detection_model", help="type of used detection model:"
                                  " yolov5n, yolov5s, yolov5m and yolov5l are allowed",
        default='yolov5n', type=str)
    parser.add_argument(
        "--reid_model", help="type of used REID model: "
                             "osnet_x1_0, osnet_x0_75, osnet_x0_5, and osnet_x0_25 are allowed",
        default='osnet_x0_75', type=str)
    parser.add_argument(
        "--segmentation", help="flag to apply segmentation to the video",
        default=False, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir,
        args.detection_file,
        args.output_file,
        args.min_confidence,
        args.nms_max_overlap,
        args.min_detection_height,
        args.max_cosine_distance,
        args.nn_budget,
        args.display,
        args.detection_model,
        args.reid_model,
        args.segmentation
    )
