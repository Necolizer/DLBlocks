import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from PIL import Image
import os


# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)
    return images


# Function to load texts from a file
def load_texts_from_file(filepath):
    with open(filepath, 'r') as f:
        texts = [line.strip() for line in f.readlines()]
    return texts


# Function to compute FID, Inception Score, and CLIP Score
def compute_metrics(real_images, gen_images, prompt_texts, device):
    # Initialize transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Prepare real and generated images as tensors
    real_tensors = torch.stack([transform(img) for img in real_images]).to(device)
    gen_tensors = torch.stack([transform(img) for img in gen_images]).to(device)

    # Compute FID
    fid_metric = FrechetInceptionDistance().to(device)
    fid_metric.update(real_tensors, real=True)
    fid_metric.update(gen_tensors, real=False)
    fid_value = fid_metric.compute()

    # Compute Inception Score
    inception_metric = InceptionScore(splits=10).to(device)
    inception_metric.update(gen_tensors)
    is_mean, is_std = inception_metric.compute()

    # Compute CLIP Score
    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)
    clip_metric.update(gen_tensors, prompt_texts)
    clip_score = clip_metric.compute()

    return fid_value.item(), is_mean.item(), is_std.item(), clip_score.item()


def main(real_folder, gen_folder, text_file, device='cpu'):
    # Load images and texts
    real_images = load_images_from_folder(real_folder)
    gen_images = load_images_from_folder(gen_folder)
    prompt_texts = load_texts_from_file(text_file)

    assert len(real_images) > 0, "No real images found"
    assert len(gen_images) > 0, "No generated images found"
    assert len(prompt_texts) == len(gen_images), "Number of texts must match number of generated images"

    # Compute FID, Inception Score, and CLIP Score
    fid, is_mean, is_std, clip_score = compute_metrics(real_images, gen_images, prompt_texts, device=device)
    
    print(f'FID: {fid}')
    print(f'Inception Score: {is_mean} ± {is_std}')
    print(f'CLIP Score: {clip_score}')


if __name__ == '__main__':
    real_folder = 'path_to_real_images'
    gen_folder = 'path_to_generated_images'
    text_file = 'path_to_text_file'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(real_folder, gen_folder, text_file, device)
