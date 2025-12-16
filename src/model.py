import torch
import torch.nn as nn
from torchvision import models

class DeepFakeMobileNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepFakeMobileNet, self).__init__()
        
        # 1. Load the pre-trained MobileNetV3 Small model
        # "weights='DEFAULT'" loads the model with ImageNet pre-trained weights
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v3_small(weights=weights)
        
        # 2. Modify the classifier for binary classification
        # MobileNetV3 Small ends with 576 features before the final layer.
        # We only want 1 output neuron (Probability of being Fake).
        
        
        # We get the original classifier
        original_classifier = self.model.classifier
        
        # We change the last layer (index 3)
        # Input: 1024 (size of ), Output: 1
        original_classifier[3] = nn.Linear(in_features=1024, out_features=1)
        
        self.model.classifier = original_classifier
        
        

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """Freeze the feature extractor for initial training"""
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the feature extractor for fine-tuning"""
        for param in self.model.features.parameters():
            param.requires_grad = True