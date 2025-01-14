import os
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm
from multiprocessing import Pool
from torch.cuda.amp import autocast
import time
import os
from sklearn.metrics.pairwise import pairwise_distances


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Set GPU memory usage limit to 90%
torch.cuda.set_per_process_memory_fraction(0.9, device=torch.device("cuda:0"))

# Define a custom dataset to read images from a directory
class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_paths = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)  # Load image with OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
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



def noris_sum_batched(unlabeled_loader, feature_extractor, total_images, lambda_param, batch_size=16, device="cuda"):
    # Step 1: Extract features
    features, image_paths = [], []
    for images, paths in tqdm(unlabeled_loader, desc="Extracting features"):
        images = images.to(device)
        with autocast():  # Mixed precision for faster processing
            feats = feature_extractor(images).detach()
        features.append(feats.float())  # Ensure features are in float32
        image_paths.extend(paths)

    features = torch.cat(features, dim=0)  # Combine all features into a single tensor
    print(f"Extracted features for {features.size(0)} images.")

    num_features = features.size(0)

    # Step 2: Precompute pairwise distances in batches
    distances = torch.zeros((num_features, num_features), dtype=torch.float32, device=device)
    print("Computing pairwise distances in batches...")
    
    for i in tqdm(range(0, num_features, batch_size), desc="Batched distance computation"):
        end_i = min(i + batch_size, num_features)
        batch_features = features[i:end_i]  # Select batch
        for j in range(0, num_features, batch_size):
            end_j = min(j + batch_size, num_features)
            # Cast batch_features to float32 for cdist
            batch_distances = torch.cdist(batch_features.float(), features[j:end_j].float(), p=2)
            distances[i:end_i, j:end_j] = batch_distances  # Fill the corresponding block
    
    print("Pairwise distances computed.")

    # Step 3: Uncertainty Sampling
    uncertainties = torch.ones(num_features, dtype=torch.float32, device=device)  # Initialize uncertainties
    selected_indices = []

    for _ in tqdm(range(total_images), desc="Selecting images"):
        if selected_indices:
            # Compute similarity to already selected images
            sims = torch.zeros(num_features, dtype=torch.float32, device=device)
            for idx in selected_indices:
                sims += torch.exp(-distances[idx] ** 2 / lambda_param)
            gains = uncertainties - sims
        else:
            gains = uncertainties.clone()

        # Select the image with the highest gain
        best_idx = torch.argmax(gains).item()
        selected_indices.append(best_idx)

        # Update uncertainties
        sim = torch.exp(-distances[best_idx] ** 2 / lambda_param)
        uncertainties -= sim * uncertainties[best_idx]

    selected_image_paths = [image_paths[idx] for idx in selected_indices]
    return selected_image_paths

# Parallelized image copying
def copy_image(args):
    src, dest = args
    shutil.copy(src, dest)

# Main function
if __name__ == "__main__":
    # Configurations
    start_time = time.time()
    source_dir = "D:/Nilesh/labeling_work_24_12_23/yolodata/SW/20241219/original_data/SW_20241219/jpg"
    output_dir = "selected_images"
    os.makedirs(output_dir, exist_ok=True)

    total_images = 10000  # Total images to select
    lambda_param = 1.0
    batch_size = 8  # Smaller batch size for the dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    resnet = models.resnet18(pretrained=True).to(device)
    feature_extractor = FeatureExtractor(resnet).to(device)
    feature_extractor.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(source_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    selected_images = noris_sum_batched(
        unlabeled_loader=dataloader,
        feature_extractor=feature_extractor,
        total_images=total_images,
        lambda_param=lambda_param
    )

    # Copy selected images in parallel
    with Pool(processes=8) as pool:
        pool.map(copy_image, [(src, os.path.join(output_dir, os.path.basename(src))) for src in selected_images])

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total duration: {duration:.2f} seconds.")
    print("Image selection and transfer complete.")
