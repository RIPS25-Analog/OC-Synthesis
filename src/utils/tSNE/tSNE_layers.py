import os, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, models, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle

# --- Config ---
dirs = ["/home/data/pace/pace_v3_video-shuffled/train/images",
        "/home/data/processed/cnp-pace/pace_v3_randomized_20k/images",
        "/home/data/processed/cnp-pace/BG20k_pace_v3-unsplit/images",
        "/home/data/diffusion_v3_randomized/train/images",
        "/home/data/processed/3d_RP_shuffled/train/images",
        "/home/data/processed/3d_CopyPaste_shuffled/train/images"
        ]
dataset_names = ["Real", "Cut-Paste", "Cut-Paste BG20k", "Diffusion", "3D-RP", "3D-CopyPaste"]
layer_names = ["Early (conv1)", "Low-level (layer1)", "Mid-level (layer2)", "High-level (layer3)", "Final (layer4)"]
N, BATCH, SEED = 20000, 64, 0
PCA_components = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES_CACHE_DIR = "features_cache"
# ---------------

# Create cache directory if it doesn't exist
os.makedirs(FEATURES_CACHE_DIR, exist_ok=True)

tfm = transforms.Compose([
    transforms.Resize((224*3,224*3)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Custom dataset for flat image directories
class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Get all image files
        for file in os.listdir(root_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                self.image_paths.append(os.path.join(root_dir, file))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

# Multi-layer feature extractor
class MultiLayerFeat(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        
    def forward(self, x):
        features = {}
        
        # Early features (after conv1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = self.avgpool(x).view(x.size(0), -1)
        
        x = self.maxpool(x)
        
        # Layer features
        x = self.layer1(x)
        features['layer1'] = self.avgpool(x).view(x.size(0), -1)
        
        x = self.layer2(x)
        features['layer2'] = self.avgpool(x).view(x.size(0), -1)
        
        x = self.layer3(x)
        features['layer3'] = self.avgpool(x).view(x.size(0), -1)
        
        x = self.layer4(x)
        features['layer4'] = self.avgpool(x).view(x.size(0), -1)
        
        return features

# Load datasets
datasets_list = [FlatImageDataset(d, transform=tfm) for d in dirs]
subsets = []
for ds in datasets_list:
    idx = random.sample(range(len(ds)), min(N,len(ds)))
    subsets.append(Subset(ds, idx))

loaders = [DataLoader(s, BATCH, shuffle=False) for s in subsets]

# Feature extractor
resnet = models.resnet50(weights='DEFAULT')
model = MultiLayerFeat(resnet).to(device).eval()

def get_cache_filename(dataset_idx, layer_key, n_samples):
    """Generate cache filename based on dataset, layer, and number of samples"""
    dataset_name = dataset_names[dataset_idx].replace(" ", "_").replace("-", "_")
    return os.path.join(FEATURES_CACHE_DIR, f"{dataset_name}_{layer_key}_{n_samples}.pkl")

def save_features(features, dataset_idx, layer_key, n_samples):
    """Save features to cache file"""
    cache_file = get_cache_filename(dataset_idx, layer_key, n_samples)
    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"Saved features to {cache_file}")

def load_features(dataset_idx, layer_key, n_samples):
    """Load features from cache file if it exists"""
    cache_file = get_cache_filename(dataset_idx, layer_key, n_samples)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded features from {cache_file}")
        return features
    return None

def get_layer_feats(loader, layer_key, dataset_idx):
    # Try to load from cache first
    cached_features = load_features(dataset_idx, layer_key, len(loader.dataset))
    if cached_features is not None:
        return cached_features
    
    # Calculate features if not cached
    feats = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc=f"Extracting {layer_key} features", leave=False):
            layer_features = model(x.to(device))
            feats.append(layer_features[layer_key].cpu().numpy())
    
    features = np.vstack(feats)
    
    # Save to cache
    save_features(features, dataset_idx, layer_key, len(loader.dataset))
    
    return features

# Extract features for each layer
layer_keys = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']

# Create subplots for different layers
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for layer_idx, layer_key in enumerate(tqdm(layer_keys, desc="Processing layers")):
    X, y = [], []
    
    for i, loader in enumerate(tqdm(loaders, desc=f"Processing datasets for {layer_key}", leave=False)):
        f = get_layer_feats(loader, layer_key, i)
        X.append(f)
        y += [i] * len(f)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # PCA + TSNE
    n_components = min(PCA_components, X.shape[0] - 1, X.shape[1])
    X_pca = PCA(n_components).fit_transform(X)
    X_tsne = TSNE(2, init="pca", random_state=SEED).fit_transform(X_pca)
    
    # Plot
    ax = axes[layer_idx]
    for i in range(len(dirs)):
        ax.scatter(*X_tsne[y==i].T, s=3, label=dataset_names[i], alpha=0.7)
    
    ax.set_title(f'{layer_names[layer_idx]}')
    ax.legend(fontsize=8)

# Remove empty subplot
axes[-1].remove()

plt.tight_layout()
plt.savefig(f"tsne_layers-{N}_imgs.png", dpi=300, bbox_inches='tight')
plt.show()
