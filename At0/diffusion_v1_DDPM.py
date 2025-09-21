import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import math
import torchvision

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 32  # Start small for prototype
channels = 3
batch_size = 64
timesteps = 1000  # Number of diffusion steps
epochs = 50  # Adjust as needed

# U-Net for noise prediction
class UNet(nn.Module):
    def __init__(self, channels=3):
        super(UNet, self).__init__()
        self.time_dim = 128  # Embedding dimension for timestep

        self.time_emb = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),  # Match input to time_dim
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        self.enc1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Conv2d(128, 256, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Conv2d(256, 128, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.out = nn.Conv2d(64, channels, 1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq.to(t.device))
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq.to(t.device))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # Process timestep embedding
        t = t.unsqueeze(-1).type(torch.float)  # Shape: [batch_size, 1]
        t_emb = self.pos_encoding(t, self.time_dim)  # Shape: [batch_size, time_dim]
        t_emb = self.time_emb(t_emb)  # Shape: [batch_size, time_dim]

        # Encoder
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(self.pool(x1)))

        # Bottleneck
        b = F.relu(self.bottleneck(self.pool(x2)))

        # Decoder with skip connections
        d1 = F.relu(self.dec1(torch.cat([self.up1(b), x2], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([self.up2(d1), x1], dim=1)))
        return self.out(d2)

# Diffusion Model
class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super(DDPM, self).__init__()
        self.model = model
        self.timesteps = timesteps
        self.betas = self.linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward(self, x, t):
        return self.model(x, t)

    @torch.no_grad()
    def p_sample(self, x, t):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)  # Fixed to x.shape
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        if t[0] == 0:
            return model_mean
        posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape).to(device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=device)
            x = self.p_sample(x, t)
        return x

# Custom Dataset
class AstroDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
train_image_dir = r'D:\Shantanu\MBTR\DATA\Images\train'
dataset = AstroDataset(train_image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
unet = UNet(channels=channels).to(device)
ddpm = DDPM(unet, timesteps=timesteps).to(device)

# Optimizer
optimizer = torch.optim.Adam(ddpm.parameters(), lr=0.0001)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        images = batch.to(device)
        t = torch.randint(0, timesteps, (images.shape[0],)).to(device)
        noise = torch.randn_like(images)
        noisy_images = ddpm.q_sample(images, t, noise)
        predicted_noise = ddpm(noisy_images, t)
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Sample and save
    if (epoch + 1) % 10 == 0:
        samples = ddpm.sample((16, channels, image_size, image_size))
        samples = (samples.clamp(-1, 1) + 1) / 2
        grid = torchvision.utils.make_grid(samples, nrow=4)
        torchvision.utils.save_image(grid, f'ddpm_sample_epoch_{epoch+1}.png')

print("Training complete!")