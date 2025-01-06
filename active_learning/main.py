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

# Efficient NORIS-Sum implementation
def noris_sum_resnet_torch(unlabeled_loader, feature_extractor, batch_size, lambda_param):
    # Step 1: Extract features
    features, image_paths = [], []
    for images, paths in tqdm(unlabeled_loader, desc="Extracting features"):
        images = images.to(device)
        with autocast():  # Mixed precision for faster processing
            feats = feature_extractor(images).detach().cpu().numpy()
        features.extend(feats)
        image_paths.extend(paths)

    features = np.array(features)
    print(f"Extracted features for {len(features)} images.")

    # Step 2: Compute pairwise distances using torch.cdist
    features_tensor = torch.tensor(features).to(device)
    pairwise_distances = torch.cdist(features_tensor, features_tensor).cpu().numpy()

    uncertainties = np.ones(len(features))  # Dummy uncertainties, replace with actual logic
    selected_indices = []

    for _ in tqdm(range(batch_size), desc="Selecting images"):
        # Compute gains
        gains = uncertainties.copy()
        if selected_indices:
            sims = np.exp(-pairwise_distances[:, selected_indices] ** 2 / lambda_param)
            gains -= sims.dot(uncertainties[selected_indices])

        best_idx = np.argmax(gains)
        selected_indices.append(best_idx)

        # Update uncertainties vectorized
        dist = pairwise_distances[:, best_idx]
        sim = np.exp(-dist ** 2 / lambda_param)
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

    batch_size = 10000
    lambda_param = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    resnet = models.resnet18(pretrained=True).to(device)
    feature_extractor = FeatureExtractor(resnet).to(device)
    feature_extractor.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),  # OpenCV handles resizing, normalize here
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(source_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    selected_images = noris_sum_resnet_torch(
        unlabeled_loader=dataloader,
        feature_extractor=feature_extractor,
        batch_size=batch_size,
        lambda_param=lambda_param
    )

    # Copy selected images in parallel
    with Pool(processes=8) as pool:
        pool.map(copy_image, [(src, os.path.join(output_dir, os.path.basename(src))) for src in selected_images])

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total duration: {duration:.2f} seconds.")
    print("Image selection and transfer complete.")
