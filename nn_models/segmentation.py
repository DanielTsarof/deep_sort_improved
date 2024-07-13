import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

class Segmentation:
    def __init__(self):
        """
        Initialize the segmentation model.
        """
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def segment_person(self, image, save_path=None):
        """
        Perform segmentation on the entire image to extract the mask for the class person.

        Args:
        image (numpy.ndarray): The input image.
        save_path (str or None): Path to save the segmentation result. If None, the result is not saved.

        Returns:
        mask (numpy.ndarray): Binary mask for the class person.
        """
        # Apply the transformations
        input_tensor = self.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            # Perform the segmentation
            output = self.model(input_tensor)['out'][0]
            output_predictions = output.argmax(0).byte().cpu().numpy()

        # Resize the output mask to match the original image size
        mask = cv2.resize(output_predictions, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create a binary mask for the class 'person' (class index 15 in COCO dataset)
        person_mask = np.where(mask == 15, 1, 0).astype(np.uint8)

        # Save the mask if save_path is provided
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            mask_image = Image.fromarray(person_mask * 255)
            mask_image.save(save_path)

        return person_mask

    def overlay_mask(self, image, mask, color):
        """
        Overlay the mask on the input image with the given color.

        Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): Binary mask for the class person.
        color (tuple): Color for the mask overlay in BGR format.

        Returns:
        overlay (numpy.ndarray): Image with mask overlay.
        """
        overlay = image.copy()
        overlay[mask == 1] = color

        alpha = 0.5  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, overlay)

        return overlay


if __name__ == '__main__':
    # Путь к изображению
    image_path = "/home/dtsarev/master_of_cv/sem2/deep_learning_in_cv/dl_for_cv_2021/final_project/deep_sort_improved/data/MOT16/test/MOT16-06/img1/000047.jpg"
    image = cv2.imread(image_path)

    # Пример bounding box
    bbox = (159, 137, 114, 220)  # (x, y, width, height)

    # Путь для сохранения маски
    save_path = "./path_to_save_mask.png"

    # Создание экземпляра класса сегментации
    segmenter = Segmentation()

    # Сегментация
    mask = segmenter.segment_person(image, bbox, save_path)

    # Цвет для наложения маски (например, красный)
    color = (0, 0, 255)  # BGR format

    # Наложение маски на изображение
    overlayed_image = segmenter.overlay_mask(image, mask, color)

    # Отображение результата
    cv2.imshow("Overlayed Image", overlayed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()