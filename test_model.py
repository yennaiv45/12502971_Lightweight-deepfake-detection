import torch
from src.model import DeepFakeMobileNet

def test_architecture():
    # 1. Instancier le modèle
    model = DeepFakeMobileNet(pretrained=False)
    print("✅ Modèle MobileNetV3 chargé.")
    
    # 2. Créer une fausse image (Batch de 2 images, 3 canaux RGB, 224x224)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # 3. Passer l'image dans le modèle (Forward pass)
    output = model(dummy_input)
    
    # 4. Vérifier la sortie
    print(f"Forme de la sortie : {output.shape}")
    
    # On attend [2, 1] : 2 prédictions (une par image du batch), 1 valeur par prédiction
    if output.shape == (2, 1):
        print("✅ Test réussi : Le modèle accepte les images et sort une prédiction binaire.")
    else:
        print(f"❌ Erreur : Forme inattendue {output.shape}")

if __name__ == "__main__":
    test_architecture()