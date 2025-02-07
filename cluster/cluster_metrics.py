import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
from torch.cuda.amp import autocast  # For mixed precision
import shutil
from collections import defaultdict

# Feature extraction dataset class
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


# Function to calculate clustering metrics
def evaluate_clustering(features_reduced, cluster_labels):
    metrics = {}

    # Silhouette Score
    metrics['silhouette_score'] = silhouette_score(features_reduced, cluster_labels)

    # Davies-Bouldin Index
    metrics['davies_bouldin_index'] = davies_bouldin_score(features_reduced, cluster_labels)

    # Cluster counts
    unique, counts = np.unique(cluster_labels, return_counts=True)
    metrics['clusters_count'] = dict(zip(unique, counts))

    return metrics


# Function to calculate diversity within a cluster
def calculate_diversity(features, cluster_labels):
    diversity_scores = {}
    clusters = defaultdict(list)

    # Group features by cluster labels
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(features[idx])

    # Calculate diversity (average pairwise distance) for each cluster
    for label, cluster_features in clusters.items():
        if len(cluster_features) > 1:
            distances = pairwise_distances(cluster_features)
            diversity_scores[label] = np.mean(distances)
        else:
            diversity_scores[label] = 0  # No diversity in single-item clusters

    return diversity_scores


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    image_dir = f"D:/Nilesh/diverse_image_sampling/images"
    
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    dataset = ImageDataset(image_paths, transform=transform)
    batch_size = 128 if torch.cuda.is_available() else 32  # Adjust batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Allocate memory for features
    num_images = len(image_paths)
    feature_dim = 2048  # ResNet50 last layer feature size
    features = np.zeros((num_images, feature_dim), dtype=np.float32)
    image_paths_list = []

    # Mixed precision for faster feature extraction
    with torch.no_grad():
        index = 0
        for batch_images, batch_paths in tqdm(dataloader, desc="Extracting Features"):
            batch_images = batch_images.to(device)
            with autocast():
                batch_features = model(batch_images).squeeze(-1).squeeze(-1).cpu().numpy()
            batch_size_current = batch_features.shape[0]
            features[index:index + batch_size_current] = batch_features
            image_paths_list.extend(batch_paths)
            index += batch_size_current

    # 2. Dimensionality Reduction
    print("Reducing Dimensionality...")
    pca = PCA(n_components=50)
    features_reduced = pca.fit_transform(features)

    # 3. Clustering
    n_clusters = 50
    print("Clustering Features...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_reduced)

    # Clustering evaluation
    clustering_metrics = evaluate_clustering(features_reduced, cluster_labels)
    print("Clustering Metrics:", clustering_metrics)

    # Diversity scores
    diversity_scores = calculate_diversity(features, cluster_labels)
    print("Diversity Scores (sample):", dict(list(diversity_scores.items())[:5]))  # Display a sample

    # 4. Sampling with Limit on Similar Images
    max_per_cluster = 10  # Maximum similar images allowed per cluster
    cluster_counts = {i: 0 for i in range(n_clusters)}
    selected_image_paths = []

    for idx, label in enumerate(cluster_labels):
        if cluster_counts[label] < max_per_cluster:
            selected_image_paths.append(image_paths_list[idx])
            cluster_counts[label] += 1

    print('Selected image paths', len(selected_image_paths))

    # 5. Async Saving of Selected Images
    output_dir = f"D:/Nilesh/diverse_image_sampling/selected_images"
    
    os.makedirs(output_dir, exist_ok=True)

    async def save_image_async(path):
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)
        shutil.copy(path, output_path)

    async def save_all_images_async(image_paths):
        tasks = [save_image_async(path) for path in image_paths]
        await asyncio.gather(*tasks)

    print("Saving Selected Images...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(save_all_images_async(selected_image_paths))

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total duration: The process took {duration:.2f} seconds.")
    print(f"Selected {len(selected_image_paths)} images saved to {output_dir}.")


if __name__ == '__main__':
    main()
