import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np
import os
import time


from src.dataset import DeepfakeDataset
from src.model import DeepFakeMobileNet

# hyperparameters
BATCH_SIZE = 32          
LEARNING_RATE = 1e-4     # little learning rate for fine-tuning
EPOCHS = 10              
NUM_WORKERS = 0          # 0 for Windows, otherwise set to number of CPU cores
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Accelerated training with mixed precision
        if DEVICE.type == 'cuda':
            # Warning correcte : torch.cuda.amp.autocast is only available on CUDA devices
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(probs)
        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(loader)
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5 
        
    epoch_acc = accuracy_score(all_labels, np.round(all_preds))
    
    return avg_loss, epoch_acc, epoch_auc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs)
            
    avg_loss = running_loss / len(loader)
    try:
        val_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        val_auc = 0.5
    val_acc = accuracy_score(all_labels, np.round(all_preds))
    
    return avg_loss, val_acc, val_auc

def main():
    # We apply data augmentations only on Train to avoid "memorization"
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        
        # 1.Miroir horizontal (50% chance)
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 2. Light rotation (-15 to +15 degrés)
        transforms.RandomRotation(degrees=15),
        
        # 3. Changing of brightness/contrast/saturation/hue
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        
        # 4. Gaussian Blur (20% chance)
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation : We keep it simple, no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # VIDEO-LEVEL SPLIT
    print("We load the datasets...")
    
    # We explicitly load 'train' and 'val' folders
    train_dataset = DeepfakeDataset(root_dir="data/processed", split='train', transform=train_transform)
    val_dataset = DeepfakeDataset(root_dir="data/processed", split='val', transform=val_transform) 

    if len(train_dataset) == 0:
        print("Error the dataset train is empty.")
        return
    
    if len(val_dataset) == 0:
        print("Error the dataset Val is empty.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"Images Train : {len(train_dataset)} | Images Val : {len(val_dataset)}")

    # 3. Model
    model = DeepFakeMobileNet(pretrained=True).to(DEVICE)
    
    # 4. Loss & Optimizer
    # Weighted Binary Cross-Entropy Loss to compensate class imbalance
    pos_weight = torch.tensor([1.0]).to(DEVICE) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # We add L2 Regularization via weight_decay to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # 5. Training Loop
    best_auc = 0.0
    start_time = time.time()
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion)
        
        print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"   Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
        
        # We save the best model based on Val AUC
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")
            print(" We save the new best model.")

    total_time = time.time() - start_time
    print(f"\n Training is done in  {total_time/60:.1f} minutes.")
    print(f"🏆 Best AUC : {best_auc:.4f}")

if __name__ == "__main__":
    main()