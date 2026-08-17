import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load Data (Deep Learning Needs Color!)
# We keep the 3 color channels (RGB) this time, and resize to 64x64.
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # Normalizing helps neural networks train faster by centering the math around 0
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

print("Loading dataset for CNN...")
train_dataset = torchvision.datasets.DTD(root='./data', split='train', partition=1,transform=transform, download=True)
test_dataset = torchvision.datasets.DTD(root='./data', split='test', partition=1,transform=transform, download=True)

# Batch size of 32 is standard for training small networks
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

# 2. Define the Shallow CNN Architecture
class ShallowTextureCNN(nn.Module):
    def __init__(self, num_classes):
        super(ShallowTextureCNN, self).__init__()
        
        # Layer 1: Looks for tiny, basic patterns (edges, color blobs)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks image to 32x32
        
        # Layer 2: Combines tiny patterns into larger textures (grids, weaves)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks image to 16x16
        
        # Layer 3: Flattens the final grid and makes the final decision
        # 64 channels * 16 height * 16 width = 16384 total signals
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes) # Outputs a score for all 47 classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Setup Training Logic
def main():
    # Automatically use Apple Silicon GPU, Nvidia GPU, or default to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ShallowTextureCNN(num_classes=num_classes).to(device)
    
    # Loss function (how wrong the model is) and Optimizer (how it updates itself)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Training Loop (Epochs)
    epochs = 15 # Train for 15 full passes over the dataset
    print("\nStarting Training...")
    
    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()    # 1. Clear old math
            outputs = model(images)  # 2. Guess the textures
            loss = criterion(outputs, labels) # 3. Calculate how wrong the guess was
            loss.backward()          # 4. Calculate the corrections
            optimizer.step()         # 5. Apply the corrections to the weights
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    print("Finished Training! Saving model weights...")
    torch.save(model.state_dict(), 'texture_cnn_weights.pth')

    # 5. Evaluate the CNN
    print("\nTesting the CNN Classifier...")
    model.eval() # Set model to testing mode (no learning allowed)
    all_preds = []
    all_labels = []

    # Disable gradient math to speed up testing
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            # Find the class with the highest score
            _, predicted = torch.max(outputs.data, 1) 
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate Final Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n========================================")
    print(f"DEEP LEARNING PIPELINE ACCURACY: {accuracy * 100:.2f}%")
    print(f"========================================\n")
    
    # Generate Confusion Matrix
    print("Generating CNN Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Deep Learning Pipeline (CNN) Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()