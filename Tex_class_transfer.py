import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from eval_utils import set_seed, report

# 
# 1. High-Resolution Transforms (ImageNet Standards)
# 
# ResNet expects 224x224 images and very specific mathematical normalization
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224), # Randomly crops to 224x224 to add variety
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # Clean center crop for testing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading dataset for Transfer Learning...")
train_dataset = torchvision.datasets.DTD(root='./data', split='train', partition=1,
                                         transform=train_transform, download=True)
test_dataset = torchvision.datasets.DTD(root='./data', split='test', partition=1,
                                        transform=test_transform, download=True)

# Increased batch size to 64 to leverage the 24GB RAM limits
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

num_classes = len(train_dataset.classes)

# 
# 2. Setup the Pre-trained ResNet-50
# 
def get_model(num_classes):
    # Download the massive, pre-trained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Freeze all the early layers so we don't destroy their pre-learned knowledge!
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the very last layer (which was trained for 1000 random ImageNet objects) 
    # with a brand new, untrained layer for our 47 texture classes.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)

    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # We only pass the parameters of the final layer (model.fc) to the optimizer.
    # The rest of the network stays frozen.
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    epochs = 15 # Transfer learning converges much faster!
    print("\nStarting Transfer Learning Training...")
    
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
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    # 
    # 3. Evaluate the ResNet Model
    # 
    print("\nTesting the Transfer Learning Model...")
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

    report(all_labels, all_preds, train_dataset.classes, "ResNet-50 Transfer (bonus)")

if __name__ == '__main__':
    main()