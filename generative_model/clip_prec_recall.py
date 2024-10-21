# python clip_prec_recall.py --real_images /path/to/real_images --generated_images /path/to/generated_images

import os
import argparse
from PIL import Image
import torch
import clip
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_images_from_folder(folder_path):
    """
    Loads all images from a folder.
    :param folder_path: Path to the folder containing images.
    :return: List of PIL images.
    """
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                images.append(img.convert("RGB"))  # Ensure all images are in RGB mode
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images

def extract_clip_features(images, batch_size=32):
    """
    Extract CLIP embeddings for a set of images.
    :param images: List of images (PIL format).
    :param batch_size: Batch size for CLIP inference.
    :return: Numpy array of CLIP embeddings.
    """
    preprocessed_images = [preprocess(image).unsqueeze(0) for image in images]
    image_tensor = torch.cat(preprocessed_images).to(device)
    dataloader = DataLoader(image_tensor, batch_size=batch_size)

    features = []
    with torch.no_grad():
        for batch in dataloader:
            features.append(model.encode_image(batch).cpu().numpy())

    return np.vstack(features)

def compute_precision_recall(real_features, generated_features, k=5):
    """
    Compute Precision and Recall based on CLIP features using k-nearest neighbors.
    :param real_features: CLIP features of real images.
    :param generated_features: CLIP features of generated images.
    :param k: Number of nearest neighbors for evaluation.
    :return: Precision and Recall values.
    """
    real_features = real_features / np.linalg.norm(real_features, axis=1, keepdims=True)
    generated_features = generated_features / np.linalg.norm(generated_features, axis=1, keepdims=True)

    # Fit k-NN on real data
    knn = NearestNeighbors(n_neighbors=k).fit(real_features)

    # Precision: How many generated samples are close to real samples
    distances, _ = knn.kneighbors(generated_features)
    precision = np.mean(distances.min(axis=1) < 0.5)

    # Recall: How many real samples are captured by generated samples
    knn_gen = NearestNeighbors(n_neighbors=k).fit(generated_features)
    distances, _ = knn_gen.kneighbors(real_features)
    recall = np.mean(distances.min(axis=1) < 0.5)

    return precision, recall

def main(real_images_folder, generated_images_folder):
    # Load images from folders
    real_images = load_images_from_folder(real_images_folder)
    generated_images = load_images_from_folder(generated_images_folder)

    # Extract CLIP features for real and generated images
    real_features = extract_clip_features(real_images)
    generated_features = extract_clip_features(generated_images)

    # Compute CLIP-based Precision and Recall
    precision, recall = compute_precision_recall(real_features, generated_features)
    print(f"CLIP Precision: {precision:.4f}, CLIP Recall: {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute CLIP Precision and Recall for generated images.')
    parser.add_argument('--real_images', type=str, required=True, help='Path to the folder containing real images.')
    parser.add_argument('--generated_images', type=str, required=True, help='Path to the folder containing generated images.')

    args = parser.parse_args()

    # Call main with provided paths
    main(args.real_images, args.generated_images)
