import torch
from src.model import DeepFakeMobileNet

def test_architecture():
    # 1. Instantiate the model
    model = DeepFakeMobileNet(pretrained=False)
    print("✅ MobileNetV3 model loaded.")
    
    # 2. Create a dummy image (Batch of 2 images, 3 RGB channels, 224x224)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # 3. Pass the image through the model (Forward pass)
    output = model(dummy_input)
    
    # 4. Check the output
    print(f"Output shape: {output.shape}")
    
    # We expect [2, 1]: 2 predictions (one for each image in the batch), 1 value per prediction
    if output.shape == (2, 1):
        print("✅ Test successful: The model accepts images and returns a binary prediction.")
    else:
        print(f"❌ Error: Unexpected shape {output.shape}")

if __name__ == "__main__":
    test_architecture()