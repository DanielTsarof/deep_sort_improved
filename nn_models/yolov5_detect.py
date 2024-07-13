import torch


def load_yolo_model(weights_path):
    """
    Load YOLOv5 model from the specified path.

    Args:
    weights_path (str): Path to the YOLOv5 weights file.

    Returns:
    model: Loaded YOLOv5 model.
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    model.cuda()  # Move model to GPU
    return model


def detect_objects(model, img):
    """
    Perform object detection on an image using the provided YOLOv5 model.

    Args:
    model: YOLOv5 model.
    img (numpy.ndarray): Image array.

    Returns:
    results: Detection results.
    """
    results = model(img)
    return results


def xywh_to_ltwh(xywh):
    """
    Convert bounding box format from xywh to ltwh.

    Args:
    xywh (list): Bounding box in xywh format.

    Returns:
    ltwh (list): Bounding box in ltwh format.
    """
    x_center, y_center, width, height = xywh
    x1 = int((x_center - width / 2))
    y1 = int((y_center - height / 2))

    return [x1, y1, int(width), int(height)]
