import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.linalg import eigh
from PIL import Image
from tqdm import tqdm
import shutil
import time 


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path


def dpp(kernel_matrix, max_samples):
    eigvals, eigvecs = eigh(kernel_matrix)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    selected_indices = []
    for _ in range(max_samples):
        prob = eigvals / (eigvals + 1)
        selected = np.random.choice(len(eigvals), p=prob / sum(prob))
        selected_indices.append(selected)
        eigvals = eigvals * (1 - eigvecs[:, selected] ** 2)
    return selected_indices


def main():
    start_time = time.time()
    # Step 1: Load images
    image_folder = "RGB"  # Replace with your path
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]

    # Define transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Step 2: Initialize ResNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove the classification head
    model = model.to(device)
    model.eval()

    # Step 3: Extract features
    features = []
    file_paths = []

    print("Extracting features using ResNet...")
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)
            output = model(images)
            features.append(output.cpu().numpy())
            file_paths.extend(paths)

    features = np.concatenate(features)

    # Step 4: Cluster features with k-means
    print("Clustering features...")
    n_clusters = 1000  # Adjust this number if needed
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # Step 5: Construct the kernel matrix
    print("Constructing kernel matrix...")
    kernel_matrix = pairwise_kernels(features, metric="rbf")  # RBF kernel, can be adjusted

    # Step 6: Diversity sampling using Determinantal Point Processes (DPP)
    print("Performing diversity sampling with DPP...")
    subset_indices = dpp(kernel_matrix, max_samples=1000)
    selected_paths = [file_paths[i] for i in subset_indices]

    # Step 7: Save final subset
    output_folder = "selected_images_1000_dpp"
    os.makedirs(output_folder, exist_ok=True)
    print("Saving selected images...")
    for path in tqdm(selected_paths, desc="Saving images"):
        filename = os.path.basename(path)
        output_path = os.path.join(output_folder, filename)
        shutil.copy(path, output_path)  # Creates a symlink to save space
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total duration: The process took {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
