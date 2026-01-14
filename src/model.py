import torch
import torch.nn as nn
from torchvision import models

class DeepFakeMobileNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepFakeMobileNet, self).__init__()
        
        # 1. Load the MobileNetV3 Small model, optionally with pre-trained weights
        # weights='DEFAULT' loads weights pre-trained on ImageNet
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v3_small(weights=weights)
        
        # 2. Modify the final layer for our binary classification task.
        # The original classifier for MobileNetV3 Small is a sequence of layers.
        # The layer before the last one outputs 1024 features. We need to map this to a single output logit.
        
        # Get the original classifier sequence
        original_classifier = self.model.classifier
        
        # Replace the last layer (a Linear layer at index 3)
        # Input features: 1024, Output features: 1 (a single logit for 'fake' vs 'real')
        original_classifier[3] = nn.Linear(in_features=1024, out_features=1)
        
        self.model.classifier = original_classifier

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """Freezes the convolutional layers (feature extractor) to train only the classifier."""
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the convolutional layers to allow for end-to-end fine-tuning."""
        for param in self.model.features.parameters():
            param.requires_grad = True