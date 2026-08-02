import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

class DeepfakeDetector(nn.Module):

    def __init__(self):

        super().__init__()
        
        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        
        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            2
        )
        
    def forward(self, x):
        return self.model(x)
        
if __name__ == "__main__":

    import torch

    model = DeepfakeDetector()

    dummy = torch.randn(1, 3, 224, 224)

    output = model(dummy)

    print("Output Shape:", output.shape)