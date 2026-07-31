import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from skimage.filters import gabor_kernel
from scipy import ndimage
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load the Data (Same as Step 1)
# ---------------------------------------------------------
# We convert images to grayscale because Gabor filters look at structure, not color.
transform = transforms.Compose([
    transforms.Resize((64, 64)), # Shrinking slightly to speed up classical math
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

print("Loading dataset...")
train_dataset = torchvision.datasets.DTD(root='./data', split='train', transform=transform)
test_dataset = torchvision.datasets.DTD(root='./data', split='test', transform=transform)

# ---------------------------------------------------------
# 2. Build the Gabor Filter Bank (The "Hand-Crafted" part)
# ---------------------------------------------------------
# We create a bank of mathematical filters at different frequencies and angles.
def build_filters():
    filters = []
    # Test 4 different angles (e.g., horizontal stripes, vertical stripes, diagonals)
    for theta in range(4):
        theta = theta / 4. * np.pi
        # Test 2 different frequencies (thick stripes vs thin stripes)
        for frequency in (0.1, 0.4):
            kernel = np.real(gabor_kernel(frequency, theta=theta))
            filters.append(kernel)
    return filters

# ---------------------------------------------------------
# 3. Feature Extraction Function
# ---------------------------------------------------------
# This function rubs every single filter over an image and calculates a summary
def extract_features(image_tensor, filters):
    # Convert PyTorch tensor to standard numpy array
    image = image_tensor.numpy().squeeze() 
    
    features = []
    for kernel in filters:
        # Mathematically convolve the filter with the image
        filtered_image = ndimage.convolve(image, kernel, mode='wrap')
        # Calculate the mean and variance (our "summary" of the texture)
        features.append(filtered_image.mean())
        features.append(filtered_image.var())
    return np.array(features)

def process_dataset(dataset, filters):
    X = []
    y = []
    print(f"Extracting features from {len(dataset)} images. This may take a minute...")
    for i in range(len(dataset)):
        image, label = dataset[i]
        features = extract_features(image, filters)
        X.append(features)
        y.append(label)
        
        if (i+1) % 200 == 0:
            print(f"Processed {i+1} images...")
            
    return np.array(X), np.array(y)

# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
def main():
    filters = build_filters()
    
    # Extract features for Training Data
    print("\n--- Processing Training Data ---")
    X_train, y_train = process_dataset(train_dataset, filters)
    
    # Extract features for Testing Data
    print("\n--- Processing Testing Data ---")
    X_test, y_test = process_dataset(test_dataset, filters)
    
    # ---------------------------------------------------------
    # 5. Train and Evaluate the SVM
    # ---------------------------------------------------------
    print("\nTraining the SVM Classifier...")
    svm_model = SVC(kernel='rbf', C=10) # RBF is a standard, robust kernel
    svm_model.fit(X_train, y_train)
    
    print("Testing the SVM Classifier...")
    y_pred = svm_model.predict(X_test)
    
    # Calculate Final Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n========================================")
    print(f"CLASSICAL PIPELINE ACCURACY: {accuracy * 100:.2f}%")
    print(f"========================================\n")
    
    # Generate Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    
    # Plot it
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Classical Pipeline (Gabor + SVM) Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()