import os
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn


class PigFarmDatasetProcessor:
    def __init__(self, dataset_dir, output_dir, subset_size=10000, empty_image_quota=0.1):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.subset_size = subset_size
        self.empty_image_quota = empty_image_quota
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_resnet_model()
        self.transform = self._image_transform()
        self.image_paths = []
        self.embeddings = []
        self.empty_images = []
        self.non_empty_images = []
        
    def _load_resnet_model(self):
        """Load a pretrained ResNet model for feature extraction."""
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()  # Remove the final classification layer
        model = model.to(self.device)
        model.eval()
        return model

    def _image_transform(self):
        """Define the image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_images(self):
        """Load all image paths."""
        self.image_paths = [os.path.join(self.dataset_dir, fname)
                            for fname in os.listdir(self.dataset_dir)
                            if fname.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Loaded {len(self.image_paths)} images.")

    def _detect_pigs(self, image):
        """Dummy pig detection (placeholder for actual object detection logic)."""
        # A more robust approach should use a pre-trained object detection model like YOLO or Faster-RCNN.
        # For simplicity, we'll simulate pig detection by checking pixel variance.
        pixel_variance = np.var(np.array(image))
        return pixel_variance > 10  # Example heuristic

    def classify_images(self):
        """Classify images into empty and non-empty."""
        for image_path in self.image_paths:
            image = Image.open(image_path).convert('RGB')
            if self._detect_pigs(image):
                self.non_empty_images.append(image_path)
            else:
                self.empty_images.append(image_path)
        print(f"Classified {len(self.empty_images)} empty and {len(self.non_empty_images)} non-empty images.")

    def extract_features(self):
        """Extract features from non-empty images using ResNet."""
        embeddings = []
        for image_path in self.non_empty_images:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(image_tensor).squeeze(0).cpu().numpy()
            embeddings.append(embedding)
        self.embeddings = np.array(embeddings)
        print(f"Extracted features for {len(self.embeddings)} non-empty images.")

    def cluster_images(self, num_clusters=100):
        """Perform clustering on image embeddings."""
        pca = PCA(n_components=50)  # Reduce dimensionality for clustering
        reduced_embeddings = pca.fit_transform(self.embeddings)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(reduced_embeddings)
        print(f"Clustered images into {num_clusters} clusters.")

    def sample_images(self):
        """Sample images from each cluster."""
        sampled_images = []
        cluster_to_images = {i: [] for i in range(max(self.cluster_labels) + 1)}
        for idx, label in enumerate(self.cluster_labels):
            cluster_to_images[label].append(self.non_empty_images[idx])

        for cluster, images in cluster_to_images.items():
            num_samples = min(len(images), self.subset_size // len(cluster_to_images))
            sampled_images.extend(random.sample(images, num_samples))
        print(f"Sampled {len(sampled_images)} diverse images.")
        return sampled_images

    def include_empty_images(self, sampled_images):
        """Add empty images to the final subset based on the quota."""
        num_empty_to_add = int(self.subset_size * self.empty_image_quota)
        sampled_empty_images = random.sample(self.empty_images, min(num_empty_to_add, len(self.empty_images)))
        final_images = sampled_images + sampled_empty_images
        print(f"Included {len(sampled_empty_images)} empty images. Total images: {len(final_images)}.")
        return final_images

    def save_subset(self, final_images):
        """Save the final subset of images to the output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        for image_path in final_images:
            image_name = os.path.basename(image_path)
            image = Image.open(image_path)
            image.save(os.path.join(self.output_dir, image_name))
        print(f"Saved {len(final_images)} images to {self.output_dir}.")

    def run_pipeline(self):
        """Run the entire pipeline."""
        self._load_images()
        self.classify_images()
        self.extract_features()
        self.cluster_images()
        sampled_images = self.sample_images()
        final_images = self.include_empty_images(sampled_images)
        self.save_subset(final_images)


# Usage example
if __name__ == "__main__":
    dataset_dir = "/path/to/dataset"
    output_dir = "/path/to/output"
    processor = PigFarmDatasetProcessor(dataset_dir, output_dir)
    processor.run_pipeline()
