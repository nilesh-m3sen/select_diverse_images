import os
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import time

# Define a custom dataset to read images from a directory
class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
        if self.transform:
            image = self.transform(image)
        return image, image_path

# Feature extractor using pretrained ResNet
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])  # Remove final FC layer

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            return features.squeeze()
        
# Distance and Similarity Metrics
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def gaussian_similarity(distance, lambda_param):
    return np.exp(-distance**2 / lambda_param)

def linear_similarity(distance, lambda_param):
    return max(0, 1 - distance / lambda_param)


def compute_gains_parallel(i, features, selected_indices, uncertainties, pairwise_distances, lambda_param):
    """
    Compute the gain for a single feature index in parallel.
    """
    gain = uncertainties[i]
    if selected_indices:
        sims = np.exp(-pairwise_distances[i, selected_indices] ** 2 / lambda_param)  # Gaussian similarity
        gain -= sims.dot(uncertainties[selected_indices])
    return gain, i


def noris_sum_resnet_chunked(unlabeled_loader, feature_extractor, batch_size, distance_metric, similarity_func, lambda_param, chunk_size):
    """
    Implements NORIS-Sum with chunked feature loading and parallel processing.
    """
    # Step 1: Extract features in chunks
    features, image_paths = [], []
    for images, paths in tqdm(unlabeled_loader, desc="Extracting features"):
        images = images.to(device)
        feats = feature_extractor(images).cpu().numpy()
        features.extend(feats)
        image_paths.extend(paths)

    features = np.array(features)
    pairwise_distances = cdist(features, features, metric='euclidean')
    uncertainties = np.ones(len(features))  # Dummy uncertainties, replace with actual logic
    selected_indices = []

    # Step 2: NORIS-Sum with parallel gain computation
    for _ in tqdm(range(batch_size), desc="Selecting images"):
        # Compute gains in parallel
        results = Parallel(n_jobs=-1)(
            delayed(compute_gains_parallel)(
                i, features, selected_indices, uncertainties, pairwise_distances, lambda_param
            )
            for i in range(len(features))
            if i not in selected_indices
        )
        gains, indices = zip(*results)
        best_idx = indices[np.argmax(gains)]
        selected_indices.append(best_idx)

        # Update uncertainties
        for i in range(len(features)):
            if i not in selected_indices:
                dist = pairwise_distances[i, best_idx]
                sim = np.exp(-dist ** 2 / lambda_param)
                uncertainties[i] -= sim * uncertainties[best_idx]

    selected_image_paths = [image_paths[idx] for idx in selected_indices]
    return selected_image_paths


# Main function remains unchanged with the following modifications
if __name__ == "__main__":
    # Configurations
    start_time = time.time()
    source_dir = "images"
    output_dir = "selected_images"
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 500
    lambda_param = 1.0
    chunk_size = 10000  # Process features in chunks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    resnet = models.resnet18(pretrained=True).to(device)
    feature_extractor = FeatureExtractor(resnet).to(device)
    feature_extractor.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(source_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    selected_images = noris_sum_resnet_chunked(
        unlabeled_loader=dataloader,
        feature_extractor=feature_extractor,
        batch_size=batch_size,
        distance_metric=euclidean_distance,
        similarity_func=gaussian_similarity,
        lambda_param=lambda_param,
        chunk_size=chunk_size
    )

    for image_path in tqdm(selected_images, desc="Moving selected images", total=len(selected_images)):
        shutil.move(image_path, os.path.join(output_dir, os.path.basename(image_path)))
        print(f"Moved: {image_path} -> {output_dir}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total duration: {duration:.2f} seconds.")
    print("Image selection and transfer complete.")
