import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------
# 1. Advanced Augmentations
# ---------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), # Textures often lack top/bottom orientation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Simulate lighting changes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading dataset for SOTA Fine-Tuning...")
train_dataset = torchvision.datasets.DTD(root='./data', split='train', partition=1,transform=train_transform, download=True)
test_dataset = torchvision.datasets.DTD(root='./data', split='test', partition=1,transform=test_transform, download=True)

# A batch size of 64 fits perfectly in 24GB of unified memory
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

num_classes = len(train_dataset.classes)

# ---------------------------------------------------------
# 2. Modern Backbone & True Fine-Tuning
# ---------------------------------------------------------
def get_model(num_classes):
    # Load the highly optimized EfficientNetV2
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    
    # UNFREEZE the network: We allow the entire network to learn textures!
    for param in model.parameters():
        param.requires_grad = True
        
    # Replace the classification head
    # EfficientNet's classifier is a Sequential block; [1] is the final Linear layer
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # A tiny learning rate (1e-4) gently tweaks the pre-trained backbone without destroying it
    # weight_decay adds L2 regularization to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 20
    # Cosine Annealing slowly curves the learning rate down to 0 over 20 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\nStarting SOTA Fine-Tuning Training...")
    
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()    
            outputs = model(images)  
            loss = criterion(outputs, labels) 
            loss.backward()          
            optimizer.step()         
            
            running_loss += loss.item()
            
        # Step the scheduler at the end of each epoch
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - LR: {current_lr:.6f}")

    # ---------------------------------------------------------
    # 3. Evaluate the Model
    # ---------------------------------------------------------
    print("\nTesting the SOTA Model...")
    model.eval() 
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) 
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n========================================")
    print(f"STATE-OF-THE-ART ACCURACY: {accuracy * 100:.2f}%")
    print(f"========================================\n")
    
    print("Generating Final Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("EfficientNetV2-S Fine-Tuning Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()