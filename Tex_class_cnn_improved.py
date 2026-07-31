import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------
# 1. UPGRADE: Data Augmentation
# ---------------------------------------------------------
# We randomly alter the images during training so the network never sees the exact same image twice.
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(), # Randomly flip left/right
    transforms.RandomVerticalFlip(),   # Randomly flip up/down
    transforms.RandomRotation(15),     # Randomly tilt the image
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# Test data should NEVER be augmented. Keep it pristine.
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

print("Loading dataset with Data Augmentation...")
train_dataset = torchvision.datasets.DTD(root='./data', split='train', transform=train_transform)
test_dataset = torchvision.datasets.DTD(root='./data', split='test', transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

# ---------------------------------------------------------
# 2. UPGRADE: Architecture with Dropout
# ---------------------------------------------------------
class ImprovedTextureCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedTextureCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Added Batch Normalization to stabilize learning
        self.bn1 = nn.BatchNorm2d(32) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.relu3 = nn.ReLU()
        
        # UPGRADE: Dropout randomly zeroes out 50% of neurons during training
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(512, num_classes) 

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1) 
        x = self.dropout(self.relu3(self.fc1(x))) # Apply dropout before final decision
        x = self.fc2(x)
        return x

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # We added weight_decay (L2 Regularization) to the optimizer. 
    # This mathematically penalizes the network if any single weight gets too large.
    model = ImprovedTextureCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 

    # ---------------------------------------------------------
    # 3. Training Loop (More Epochs because learning is harder now)
    # ---------------------------------------------------------
    epochs = 25 
    print("\nStarting Improved Training...")
    
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

    # ---------------------------------------------------------
    # 4. Evaluate the Improved CNN
    # ---------------------------------------------------------
    print("\nTesting the Improved CNN...")
    model.eval() # CRITICAL: This turns off Dropout during testing!
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
    print(f"IMPROVED CNN ACCURACY: {accuracy * 100:.2f}%")
    print(f"========================================\n")
    
    print("Generating Final Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Improved CNN Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()