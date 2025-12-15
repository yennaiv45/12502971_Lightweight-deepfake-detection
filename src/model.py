import torch
import torch.nn as nn
from torchvision import models

class DeepFakeMobileNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepFakeMobileNet, self).__init__()
        
        # 1. Charger le backbone pré-entraîné (Transfer Learning)
        # "weights='DEFAULT'" charge les poids appris sur ImageNet (très important pour converger vite)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v3_small(weights=weights)
        
        # 2. Modifier la tête de classification (Classifier Head)
        # MobileNetV3 Small sort 576 features avant la couche finale.
        # Nous voulons 1 seule sortie (Probabilité Fake) au lieu de 1000 classes ImageNet.
        
        # On récupère la structure originale du classifier
        original_classifier = self.model.classifier
        
        # On remplace la dernière couche linéaire (la couche 3)
        # Input: 1024 (taille de la couche précédente dans MobileNetV3-Small), Output: 1
        original_classifier[3] = nn.Linear(in_features=1024, out_features=1)
        
        self.model.classifier = original_classifier
        
        # Note : On n'ajoute pas de Sigmoid ici car on utilisera BCEWithLogitsLoss
        # qui est plus stable numériquement.

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """Gèle les poids du backbone pour ne pas les modifier au début"""
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Dégèle tout pour le fine-tuning final"""
        for param in self.model.features.parameters():
            param.requires_grad = True