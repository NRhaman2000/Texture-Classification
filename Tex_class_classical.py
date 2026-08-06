import torchvision
import torchvision.transforms as transforms
import numpy as np
from skimage.filters import gabor_kernel
from scipy import ndimage
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from skimage.feature import local_binary_pattern

from eval_utils import set_seed, report, plot_gabor_bank

# ---------------------------------------------------------
# 1. Load the Data (Same as Step 1)
# ---------------------------------------------------------
# We convert images to grayscale because Gabor filters look at structure, not color.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

print("Loading dataset...")
train_dataset = torchvision.datasets.DTD(root='./data', split='train', partition=1,transform=transform, download=True)
test_dataset = torchvision.datasets.DTD(root='./data', split='test', partition=1,transform=transform, download=True)

# ---------------------------------------------------------
# 2. Build the Gabor Filter Bank (The "Hand-Crafted" part)
# ---------------------------------------------------------
# We create a bank of mathematical filters at different frequencies and angles.

def build_gabor_bank(orientations=6, frequencies=(0.15, 0.25, 0.35, 0.45)):
    """6 angles x 4 stripe-widths = 24 filters, kept as matched PAIRS."""
    bank = []
    for o in range(orientations):
        theta = o / orientations * np.pi      # the angle the stripes run
        for f in frequencies:                 # how tightly packed the stripes are
            bank.append(gabor_kernel(f, theta=theta))
    return bank

# ---------------------------------------------------------
# 3. Feature Extraction Function
# ---------------------------------------------------------
# This function runs every single filter over an image and calculates a summary
def gabor_energy(image, bank):
    """Measure the STRENGTH of each filter's response, not its average."""
    feats = []
    for k in bank:
        r = ndimage.convolve(image, np.real(k), mode='reflect')   # one half of the pair
        i = ndimage.convolve(image, np.imag(k), mode='reflect')   # the other half
        mag = np.sqrt(r * r + i * i)          # combine: how strongly this pattern appears
        feats += [mag.mean(), mag.var()]
    return feats


def fourier_features(image, n_rings=6, n_wedges=6):
    """Slice the frequency map into rings (scale) and wedges (direction)."""
    P = np.fft.fftshift(np.abs(np.fft.fft2(image)) ** 2)   # the power spectrum
    h, w = P.shape; cy, cx = h // 2, w // 2                # find the centre
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2); r = r / (r.max() + 1e-9)  # distance out
    th = np.arctan2(Y - cy, X - cx) % np.pi                # angle around
    feats = []
    for k in range(n_rings):                  # each ring = one level of coarseness
        feats.append(P[(r >= k / n_rings) & (r < (k + 1) / n_rings)].sum())
    for j in range(n_wedges):                 # each wedge = one direction
        m = (th >= j * np.pi / n_wedges) & (th < (j + 1) * np.pi / n_wedges)
        feats.append(P[m].sum())
    f = np.asarray(feats, dtype=np.float64)
    return list(f / (f.sum() + 1e-9))         # scale to fractions so bright/dark photos match

def lbp_histogram(image, radii=((1, 8), (2, 16), (3, 24))):
    """Local Binary Patterns: describes each pixel by comparing it to its neighbours.
    Rotation-invariant and complementary to Gabor: Gabor asks "which direction do
    the stripes run", LBP asks "what local shapes appear and how often". Three
    radii capture fine, medium and coarse structure.
    """
    img = (image * 255).astype(np.uint8)          # LBP wants 0-255 integers
    feats = []
    for R, P in radii:                             # R = radius, P = neighbours sampled
        codes = local_binary_pattern(img, P, R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float64)
        feats += list(hist / (hist.sum() + 1e-9))  # normalize to a distribution
    return feats

def extract_features(image_tensor, bank):
    image = image_tensor.numpy().squeeze()
    return np.array(gabor_energy(image, bank)
                    + fourier_features(image)
                    + lbp_histogram(image))

def process_dataset(dataset, bank):
    X = []
    y = []
    print(f"Extracting features from {len(dataset)} images. This may take a minute...")
    for i in range(len(dataset)):
        image, label = dataset[i]
        features = extract_features(image, bank)
        X.append(features)
        y.append(label)
        
        if (i+1) % 200 == 0:
            print(f"Processed {i+1} images...")
            
    return np.array(X), np.array(y)

# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
def main():
    set_seed(42)
    bank = build_gabor_bank()
    plot_gabor_bank(bank)          # saves gabor_bank.png — you need this for Part C

    print("\n--- Processing Training Data ---")
    X_train, y_train = process_dataset(train_dataset, bank)
    print("\n--- Processing Testing Data ---")
    X_test, y_test = process_dataset(test_dataset, bank)

    # put every feature on a comparable scale
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTuning and training the SVM...")
    grid = GridSearchCV(
        SVC(kernel='rbf'),
        {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.001]},
        cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("best SVM params:", grid.best_params_)

    y_pred = grid.predict(X_test)
    report(y_test, y_pred, train_dataset.classes, "Classical Gabor+Fourier SVM")

if __name__ == '__main__':
    main()