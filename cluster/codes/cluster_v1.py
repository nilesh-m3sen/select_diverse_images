import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm  # Progress bar library
import time

start_time = time.time()
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Feature Extraction
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path

# Load pre-trained model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
model = model.to(device)  # Move model to GPU
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to images
image_dir = "SW_20241226_RGB"
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
dataset = ImageDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Extract features
features = []
image_paths_list = []
with torch.no_grad():
    for batch_images, batch_paths in tqdm(dataloader, desc="Extracting Features"):
        batch_images = batch_images.to(device)  # Move images to GPU
        batch_features = model(batch_images).squeeze(-1).squeeze(-1).cpu().numpy()  # Move features back to CPU
        features.append(batch_features)
        image_paths_list.extend(batch_paths)

features = np.vstack(features)  # Shape: (num_images, feature_dim)

# 2. Dimensionality Reduction
pca = PCA(n_components=50)
print("Reducing Dimensionality...")
features_reduced = pca.fit_transform(features)

# 3. Clustering
n_clusters = 1000
print("Clustering Features...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(features_reduced)

# 4. Sampling with Limit on Similar Images
selected_image_paths = []
max_per_cluster = 10  # Maximum similar images allowed per cluster
cluster_counts = {i: 0 for i in range(n_clusters)}

for idx, label in enumerate(cluster_labels):
    if cluster_counts[label] < max_per_cluster:
        selected_image_paths.append(image_paths_list[idx])
        cluster_counts[label] += 1
        
print('Selected image paths', len(selected_image_paths))


final_subset = selected_image_paths

# Save final subset in a separate folder
output_dir = "SW_20241226_RGB_selected_images"
os.makedirs(output_dir, exist_ok=True)
for path in tqdm(final_subset, desc="Saving Selected Images"):
    filename = os.path.basename(path)
    output_path = os.path.join(output_dir, filename)
    os.rename(path, output_path)

end_time = time.time()
duration = end_time - start_time
print(f"Total duration: The process took {duration:.2f} seconds.")
print(f"Selected {len(final_subset)} images saved to {output_dir}.")
