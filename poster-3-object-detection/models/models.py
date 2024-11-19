import torch.nn as nn
import torch
from torchvision import models



# Model with pretrained ResNet backbone, classification, and regression heads
class ResNetTwoHeads_old(nn.Module):
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

    def predict(self, x):
        cls, bbox_transforms = self.forward(x)
        cls_probs = torch.softmax(cls, dim=1)  # Convert logits to probabilities
        return cls, bbox_transforms, cls_probs


class ResNetTwoHeads(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.6):
        super(ResNetTwoHeads, self).__init__()

        # Backbone: Pretrained ResNet
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the original fully connected layer

        # Freeze the backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Shared fully connected layer
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classification head (pothole vs. background) with 2 additional layers
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)  # 2 classes: object and background
        )

        # Regression head for bbox transformations (tx, ty, tw, th) with 2 additional layers
        self.regressor = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 4)  # 4 values: tx, ty, tw, th
        )

    def forward(self, x):
        # Backbone features
        features = self.backbone(x)

        # Shared fully connected layer
        shared_features = self.shared_fc(features)

        # Outputs from the two heads
        cls = self.classifier(shared_features)
        bbox_transforms = self.regressor(shared_features)

        return cls, bbox_transforms

    def predict(self, x):
        cls, bbox_transforms = self.forward(x)
        cls_probs = torch.softmax(cls, dim=1)  # Convert logits to probabilities
        return cls, bbox_transforms, cls_probs