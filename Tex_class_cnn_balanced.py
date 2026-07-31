import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------
# 1. Gentle Transforms (Preserves directional textures)
# ---------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(), # Light flip only
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

print("Loading dataset for Balanced CNN...")
train_dataset = torchvision.datasets.DTD(root='./data', split='train', transform=train_transform)
test_dataset = torchvision.datasets.DTD(root='./data', split='test', transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

# ---------------------------------------------------------
# 2. 3-Layer Balanced CNN Architecture
# ---------------------------------------------------------
class BalancedTextureCNN(nn.Module):
    def __init__(self, num_classes):
        super(BalancedTextureCNN, self).__init__()
        
        # Layer 1: Low-level features (edges, colors) -> outputs 32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) 
        
        # Layer 2: Mid-level features (simple patterns) -> outputs 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        # Layer 3: High-level features (complex textures) -> outputs 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2) 

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Gentle 20% dropout
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1) 
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BalancedTextureCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Adjusted Learning Rate to 0.0003 for stable optimization
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5) 

    epochs = 20 
    print("\nStarting Training...")
    
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
    # 3. Evaluate the Balanced CNN
    # ---------------------------------------------------------
    print("\nTesting the Balanced CNN...")
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
    print(f"BALANCED CNN ACCURACY: {accuracy * 100:.2f}%")
    print(f"========================================\n")
    
    print("Generating Final Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Balanced CNN Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()