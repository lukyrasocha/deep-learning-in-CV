import torch.nn as nn
from torchvision import models



# Model with pretrained ResNet backbone, classification, and regression heads
class ResNetTwoHeads(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetTwoHeads, self).__init__()
        
        # import resnet 18 
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        #self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the original fully connected layer

        # Classification head (pothole vs. background)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)  # 2 classes: object and background
        )

        # Regression head for bbox transformations (tx, ty, tw, th)
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 4 values: tx, ty, tw, th
        )

    def forward(self, x):
        features = self.backbone(x)
        cls = self.classifier(features)
        bbox_transforms = self.regressor(features)
        return cls, bbox_transforms