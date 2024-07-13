import torchreid
import torch
from torchvision import transforms
from PIL import Image


class ReIDModel:
    def __init__(self, model_type='osnet_x0_75'):
        # Load OSNet model
        print('*** ', model_type)
        self.model = torchreid.models.build_model(
            name=model_type,
            num_classes=1000,
            pretrained=True
        ).cuda().eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract_features(self, img):
        """
        Extract features from an image using the Re-ID model.

        Args:
        img (PIL.Image): Input image.

        Returns:
        features (torch.Tensor): Extracted features.
        """
        img = self.transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            features = self.model(img)
        return features.cpu()


def load_reid_model(model_type):
    """
    Load the Re-ID model.

    Returns:
    ReIDModel: Loaded Re-ID model.
    """
    return ReIDModel(model_type)
