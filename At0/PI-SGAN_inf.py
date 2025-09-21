import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Setup logging
log_dir = r'D:/Shantanu/MBTR/DATA/logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, f'pisgan_inference_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Debug GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
else:
    print("No GPU detected; using CPU.")
logging.info(f"CUDA available: {torch.cuda.is_available()}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (must match training)
latent_dim = 100
num_classes = 8
image_size = 512

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels).view(-1, latent_dim, 1, 1)
        gen_input = torch.cat((noise, label_emb), dim=1)
        return self.main(gen_input)

# Critic (needed for consistency, though not used in inference)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 1)
        self.main = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 8, 1, 0, bias=False)
        )

    def forward(self, x, labels):
        label_emb = self.label_emb(labels).view(-1, 1, 1, 1)
        label_emb = label_emb.expand(-1, 1, x.shape[2], x.shape[3])
        crit_input = torch.cat((x, label_emb), dim=1)
        return self.main(crit_input).view(-1)

# Inference function (adapted from the original)
def inference(generator, num_samples=16, class_id=0, save_path=r'D:/Shantanu/MBTR/DATA/inference/'):
    os.makedirs(save_path, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, 1, 1, device=device)
        labels = torch.full((num_samples,), class_id, dtype=torch.long, device=device)
        fake_images = generator(noise, labels)
        fake_images = (fake_images + 1) / 2  # Scale to [0, 1]
        torchvision.utils.save_image(fake_images, os.path.join(save_path, f'inference_samples_class{class_id}.png'))
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(fake_images[i].cpu().permute(1, 2, 0))
            ax.axis('off')
        plt.savefig(os.path.join(save_path, f'inference_vis_class{class_id}.png'))
        plt.close()
        logging.info(f"Saved inference samples for class {class_id} at {save_path}")

# Load checkpoint and run inference
def load_and_infer(checkpoint_path):
    # Initialize models
    generator = Generator().to(device)
    critic = Critic().to(device)  # Included for consistency, though not used

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    # critic.load_state_dict(checkpoint['critic_state_dict'])  # Optional, not needed for inference
    logging.info(f"Loaded checkpoint from {checkpoint_path}, epoch {checkpoint['epoch']}")

    # Run inference
    for class_id in range(num_classes):  # Generate samples for all classes
        inference(generator, num_samples=16, class_id=class_id)

if __name__ == "__main__":
    # Specify the checkpoint path (e.g., latest or best)
    checkpoint_path = r'D:/Shantanu/MBTR/DATA/logs/checkpoint_epoch_80.pth'  # Adjust to your latest checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Check available files in {log_dir}")
        logging.error(f"Checkpoint not found at {checkpoint_path}")
    else:
        load_and_infer(checkpoint_path)
        print("Inference complete! Check results in D:/Shantanu/MBTR/DATA/inference/")
        logging.info("Inference complete.")