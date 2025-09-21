import os, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, models, transforms
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# --- Config ---
dirs = ["/home/data/pace/pace_v3_video-shuffled/train/images",
        "/home/data/processed/cnp-pace/pace_v3_randomized_20k/images",
        "/home/data/processed/cnp-pace/BG20k_pace_v3-unsplit/images",
        "/home/data/diffusion_v3_randomized/train/images",
        "/home/data/processed/3d_RP_shuffled/train/images",
        "/home/data/processed/3d_CopyPaste_shuffled/train/images"
        ]  # put image folders here
dataset_names = ["Real", "Cut-Paste", "Cut-Paste BG20k", "Diffusion", "3D-RP", "3D-CopyPaste"]
N, BATCH, SEED = 4000, 64, 0
PCA_components = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------

tfm = transforms.Compose([
    transforms.Resize((640,640)),
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
        return image, 0  # dummy label

# Load datasets
datasets_list = [FlatImageDataset(d, transform=tfm) for d in dirs]
subsets = []
for ds in datasets_list:
    idx = random.sample(range(len(ds)), min(N,len(ds)))
    subsets.append(Subset(ds, idx))

loaders = [DataLoader(s, BATCH, shuffle=False) for s in subsets]

# Feature extractor
class Feat(nn.Module):
    def __init__(self,m): super().__init__(); self.f = nn.Sequential(*list(m.children())[:-1])
    def forward(self,x): return self.f(x).view(x.size(0),-1)
model = Feat(models.resnet50(weights='DEFAULT')).to(device).eval()

def get_feats(loader):
    feats=[]
    with torch.no_grad():
        for x,_ in tqdm(loader, desc="Extracting features"):
            feats.append(model(x.to(device)).cpu().numpy())
    return np.vstack(feats)

X, y = [], []
for i,loader in enumerate(tqdm(loaders, desc="Processing datasets")):
    f = get_feats(loader)
    X.append(f); y += [i]*len(f)
X = np.vstack(X); y=np.array(y)

X_pca = X
# # PCA + UMAP
# print("Applying PCA...")
# n_components = min(PCA_components, X.shape[0] - 1, X.shape[1])
# X_pca = PCA(n_components).fit_transform(X)
# print("Applying UMAP...")
# X_umap = umap.UMAP(n_components=2, random_state=SEED).fit_transform(X_pca)
X_umap = umap.UMAP(n_components=2).fit_transform(X_pca)

# Plot
plt.figure(figsize=(8,6))
for i in range(len(dirs)):
    plt.scatter(*X_umap[y==i].T, s=5, label=dataset_names[i])

plt.legend()
plt.tight_layout()
plt.savefig(f"umap-{N}_imgs.png", dpi=500)
plt.show()
