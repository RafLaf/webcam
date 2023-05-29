print("importing torch")
import torch
import numpy as np

from backbone_loader.backbone_pytorch.model import (
    get_model,
)  # .resnet12 import ResNet12

print("torch imported")


class TorchBatchModelWrapper:
    """
    Wrapps a torch model to input/output ndarray
    """

    def __init__(self, model_name, weights, device="cpu"):
        self.model = get_model(model_name, weights, device=device)
        self.device = device

    def __call__(self, img):
        """
        return the features from an img
        args :
            - img(np.ndarray) : represent a batch of image (channel last convention)
        """

        assert len(img.shape) == 4

        self.model.eval()

        # convertion to tensor with channel first convention
        img = np.transpose(img, (0, 3, 1, 2))

        img = torch.from_numpy(img)
        img = img.to(self.device)

        with torch.no_grad():
            features = self.model(img)
        return features.cpu().numpy()
