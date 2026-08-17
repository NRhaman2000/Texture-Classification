import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 
# 1. Define standard transformations for the images
# 
# We resize all images to 128x128 so they are uniform.
# ToTensor() converts the image into a PyTorch mathematical matrix
# and scales the pixel values between 0 and 1.
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor() 
])

def main():
    print("Downloading and loading the DTD dataset. This might take a few minutes the first time...")
    
    # 
    # 2. Download and load the training and testing data
    # 
    # download=True tells PyTorch to fetch the images from the internet 
    # if they aren't already in the './data' folder.
    train_dataset = torchvision.datasets.DTD(
        root='./data', 
        split='train',
        download=True, 
        transform=transform
    )

    test_dataset = torchvision.datasets.DTD(
        root='./data', 
        split='test',
        download=True, 
        transform=transform
    )

    # 
    # 3. Create DataLoaders to batch the images
    # 
    # DataLoaders feed the images to our models 4 at a time.
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Success! Loaded {len(train_dataset)} training images and {len(test_dataset)} testing images.")
    print(f"Total number of texture classes: {len(train_dataset.classes)}")

    # 
    # 4. Visualization function to test if everything worked
    # 
    def imshow(img):
        img = img.numpy()
        # PyTorch stores images as [Color Channels, Height, Width]
        # Matplotlib requires [Height, Width, Color Channels], so we transpose it.
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.axis('off')
        plt.show()

    # Grab one single batch of 4 random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Show the images in a popup window
    print("\nDisplaying a sample batch of 4 images in a new window...")
    imshow(torchvision.utils.make_grid(images))
    
    # Print the text labels for those 4 images in the terminal
    class_names = [train_dataset.classes[labels[j]] for j in range(batch_size)]
    print('Labels for the displayed images: ', ' | '.join(class_names))

if __name__ == '__main__':
    main()