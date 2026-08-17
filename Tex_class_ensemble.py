import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch.nn.functional as F
import matplotlib.pyplot as plt
from eval_utils import set_seed, report
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. Shared Transforms (224x224)
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

print("Loading dataset for Ensemble Training...")
train_dataset = torchvision.datasets.DTD(root='./data', split='train', partition=1,
                                         transform=train_transform, download=True)
test_dataset = torchvision.datasets.DTD(root='./data', split='test', partition=1,
                                        transform=test_transform, download=True)

# Batch size 32 is optimal for MPS
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

# 2. Model Initialization Functions
def get_convnext(num_classes):
    print("Initializing ConvNeXt-Tiny...")
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    return model

def get_efficientnet(num_classes):
    print("Initializing EfficientNetV2-S...")
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# 3. Dedicated Training Loop
def train_model(model, model_name, device, epochs=15):
    print(f"\n--- Starting Training for {model_name} ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_batches = len(train_loader)
    for epoch in range(epochs):
        model.train() 
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()    
            outputs = model(images)  
            loss = criterion(outputs, labels) 
            loss.backward()          
            
            # Safety net to prevent 'nan' math explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()         
            
            if (i + 1) % 20 == 0 or (i + 1) == total_batches:
                print(f"[{model_name}] Epoch [{epoch+1}/{epochs}] - Batch [{i+1}/{total_batches}] - Loss: {loss.item():.4f}")
                
        scheduler.step()
    return model

# 4. Main Execution & Ensemble Evaluation
def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)

    # Initialize and train Model 1
    model_convnext = get_convnext(num_classes).to(device)
    model_convnext = train_model(model_convnext, "ConvNeXt-Tiny", device, epochs=15)
    
    # Initialize and train Model 2
    model_efficient = get_efficientnet(num_classes).to(device)
    model_efficient = train_model(model_efficient, "EfficientNetV2", device, epochs=15)

    print("\n--- Testing Dual Ensemble with Test-Time Augmentation (TTA) ---")
    model_convnext.eval()
    model_efficient.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # ConvNeXt TTA Pass 
            out_c_orig = F.softmax(model_convnext(images), dim=1)
            out_c_hflip = F.softmax(model_convnext(torch.flip(images, dims=[3])), dim=1)
            out_c_vflip = F.softmax(model_convnext(torch.flip(images, dims=[2])), dim=1)
            avg_convnext = (out_c_orig + out_c_hflip + out_c_vflip) / 3.0
            
            #  EfficientNet TTA Pass 
            out_e_orig = F.softmax(model_efficient(images), dim=1)
            out_e_hflip = F.softmax(model_efficient(torch.flip(images, dims=[3])), dim=1)
            out_e_vflip = F.softmax(model_efficient(torch.flip(images, dims=[2])), dim=1)
            avg_efficient = (out_e_orig + out_e_hflip + out_e_vflip) / 3.0
            
            # The Final Ensemble Vote 
            # Average the probabilities from both master models
            final_ensemble_outputs = (avg_convnext + avg_efficient) / 2.0
            
            _, predicted = torch.max(final_ensemble_outputs, 1) 
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    report(all_labels, all_preds, train_dataset.classes, "ConvNeXt + EfficientNetV2 Ensemble (bonus)")

if __name__ == '__main__':
    main()