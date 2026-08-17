import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn.functional as F
import matplotlib.pyplot as plt
from eval_utils import set_seed, report
from torch.optim.lr_scheduler import CosineAnnealingLR

# 
# 1. Fast & Stable Transforms (224x224)
# 
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading dataset at 224x224 resolution...")
train_dataset = torchvision.datasets.DTD(root='./data', split='train', partition=1,
                                         transform=train_transform, download=True)
test_dataset = torchvision.datasets.DTD(root='./data', split='test', partition=1,
                                        transform=test_transform, download=True)

# Batch size 32 is highly optimized for Apple Silicon MPS
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

# 
# 2. ConvNeXt-Tiny Backbone
# 
def get_model(num_classes):
    print("Downloading ConvNeXt-Tiny weights...")
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = True
        
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)

    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    epochs = 15 # 15 epochs is plenty for ConvNeXt-Tiny
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\nStarting Fast Ultimate Fine-Tuning...")
    
    total_batches = len(train_loader)
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()    
            outputs = model(images)  
            loss = criterion(outputs, labels) 
            loss.backward()          
            
            # THE SAFETY NET: Prevents the math from exploding into 'nan'
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()         
            running_loss += loss.item()
            
            # Print feedback every 15 batches so you know it's not frozen
            if (i + 1) % 15 == 0 or (i + 1) == total_batches:
                print(f"Epoch [{epoch+1}/{epochs}] - Batch [{i+1}/{total_batches}] - Current Loss: {loss.item():.4f}")
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"--- Epoch {epoch+1} Complete | Avg Loss: {running_loss/total_batches:.4f} | LR: {current_lr:.6f} ---")

    # 
    # 3. Test-Time Augmentation (TTA) Evaluation
    # 
    print("\nTesting with Test-Time Augmentation (TTA)...")
    model.eval() 
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            out_orig = F.softmax(model(images), dim=1)
            images_hflip = torch.flip(images, dims=[3])
            out_hflip = F.softmax(model(images_hflip), dim=1)
            images_vflip = torch.flip(images, dims=[2])
            out_vflip = F.softmax(model(images_vflip), dim=1)
            
            avg_outputs = (out_orig + out_hflip + out_vflip) / 3.0
            
            _, predicted = torch.max(avg_outputs, 1) 
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    report(all_labels, all_preds, train_dataset.classes, "ConvNeXt-Tiny + TTA (bonus)")

if __name__ == '__main__':
    main()