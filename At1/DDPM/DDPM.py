import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import matplotlib.pyplot as plt
import logging
from torch.amp import autocast, GradScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset-specific configuration
dataset_name = "MobilTelesco"
image_dir = f"F:/Shantanu/PISGAN/DATA/Images/{dataset_name}/train"
output_dir = f"outputs/DDPM_{dataset_name}"
logs_dir = f"logs/DDPM_{dataset_name}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{logs_dir}/ddpm_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Hyperparameters
image_size = (1536, 2048)  # Target resolution (H, W)
channels = 3
batch_size = 2  # Fits 8GB VRAM
timesteps = 1000
sampling_steps = 50  # For faster sampling
num_epochs = 100
lr = 0.0002

# Custom Dataset
class AstroDataset(Dataset):
    def __init__(self, image_dir, transform=None, use_labels=False):
        self.image_dir = image_dir
        self.transform = transform
        self.use_labels = use_labels
        exts = ('.jpg', '.jpeg', '.png')
        self.images = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.lower().endswith(exts)])
        self.labels = None if not use_labels else self._load_labels()
        logger.info(f"Loaded {len(self.images)} images from {image_dir}")

    def _load_labels(self):
        # Placeholder: Not used for DDPM
        return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.use_labels:
            label = self.labels[idx]
            return image, label
        return image

# Metrics setup
def setup_metrics(device):
    logger.info("Initializing metrics...")
    fid = FrechetInceptionDistance().to(device)
    inception = InceptionScore().to(device)
    logger.info("Metrics initialized: FID, Inception")
    return fid, inception

# Save metrics and graphs
def save_metrics_and_graphs(metrics, losses, epoch, output_dir, logs_dir):
    fid_score, inception_score = metrics
    loss, _ = losses
    logger.info(f"Epoch {epoch}, FID: {fid_score:.4f}, Inception Score: {inception_score:.4f}")
    with open(f"{logs_dir}/metrics.log", "a") as f:
        f.write(f"Epoch {epoch}, FID: {fid_score:.4f}, Inception Score: {inception_score:.4f}\n")
    plt.figure()
    plt.plot(range(epoch + 1), [loss.item() for _ in range(epoch + 1)], label="Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_graph_epoch_{epoch+1}.png")
    plt.close()
    logger.info(f"Saved loss graph to {output_dir}/loss_graph_epoch_{epoch+1}.png")

# U-Net for noise prediction
class UNet(nn.Module):
    def __init__(self, channels=3, time_dim=256):
        super(UNet, self).__init__()
        self.time_dim = time_dim
        self.time_emb = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        ).to(device)
        # Time embedding projections for each layer
        self.time_proj1 = nn.Linear(self.time_dim, 64).to(device)
        self.time_proj2 = nn.Linear(self.time_dim, 128).to(device)
        self.time_proj3 = nn.Linear(self.time_dim, 256).to(device)
        self.time_proj4 = nn.Linear(self.time_dim, 512).to(device)
        self.time_proj5 = nn.Linear(self.time_dim, 512).to(device)
        self.time_proj6 = nn.Linear(self.time_dim, 512).to(device)
        self.time_proj_b = nn.Linear(self.time_dim, 512).to(device)
        # Encoder
        self.enc1 = nn.Conv2d(channels, 64, 4, 2, 1, bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(64).to(device)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False).to(device)
        self.bn2 = nn.BatchNorm2d(128).to(device)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False).to(device)
        self.bn3 = nn.BatchNorm2d(256).to(device)
        self.enc4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False).to(device)
        self.bn4 = nn.BatchNorm2d(512).to(device)
        self.enc5 = nn.Conv2d(512, 512, 4, 2, 1, bias=False).to(device)
        self.bn5 = nn.BatchNorm2d(512).to(device)
        self.enc6 = nn.Conv2d(512, 512, 4, 2, 1, bias=False).to(device)
        self.bn6 = nn.BatchNorm2d(512).to(device)
        self.bottleneck = nn.Conv2d(512, 512, 4, 2, 1, bias=False).to(device)
        self.bn_b = nn.BatchNorm2d(512).to(device)
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False).to(device)
        self.dec1 = nn.Conv2d(1024, 512, 3, 1, 1, bias=False).to(device)  # 512 + 512
        self.bn_d1 = nn.BatchNorm2d(512).to(device)
        self.up2 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False).to(device)
        self.dec2 = nn.Conv2d(1024, 512, 3, 1, 1, bias=False).to(device)  # 512 + 512
        self.bn_d2 = nn.BatchNorm2d(512).to(device)
        self.up3 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False).to(device)
        self.dec3 = nn.Conv2d(768, 256, 3, 1, 1, bias=False).to(device)   # 256 + 512
        self.bn_d3 = nn.BatchNorm2d(256).to(device)
        self.up4 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False).to(device)
        self.dec4 = nn.Conv2d(384, 128, 3, 1, 1, bias=False).to(device)   # 128 + 256
        self.bn_d4 = nn.BatchNorm2d(128).to(device)
        self.up5 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False).to(device)
        self.dec5 = nn.Conv2d(192, 64, 3, 1, 1, bias=False).to(device)    # 64 + 128
        self.bn_d5 = nn.BatchNorm2d(64).to(device)
        self.up6 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False).to(device)
        self.dec6 = nn.Conv2d(96, 32, 3, 1, 1, bias=False).to(device)     # 32 + 64
        self.bn_d6 = nn.BatchNorm2d(32).to(device)
        self.out = nn.Conv2d(32, channels, 1).to(device)
        self.upsample = nn.Upsample(size=(1536, 2048), mode='bilinear', align_corners=False).to(device)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq.to(t.device))
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq.to(t.device))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t_emb = self.pos_encoding(t, self.time_dim)
        t_emb = self.time_emb(t_emb)
        # Project time embeddings to match each layer's channels
        t_emb1 = self.time_proj1(t_emb)[:, :, None, None]
        t_emb2 = self.time_proj2(t_emb)[:, :, None, None]
        t_emb3 = self.time_proj3(t_emb)[:, :, None, None]
        t_emb4 = self.time_proj4(t_emb)[:, :, None, None]
        t_emb5 = self.time_proj5(t_emb)[:, :, None, None]
        t_emb6 = self.time_proj6(t_emb)[:, :, None, None]
        t_emb_b = self.time_proj_b(t_emb)[:, :, None, None]
        # Encoder
        x1 = F.relu(self.bn1(self.enc1(x))) + t_emb1
        x2 = F.relu(self.bn2(self.enc2(x1))) + t_emb2
        x3 = F.relu(self.bn3(self.enc3(x2))) + t_emb3
        x4 = F.relu(self.bn4(self.enc4(x3))) + t_emb4
        x5 = F.relu(self.bn5(self.enc5(x4))) + t_emb5
        x6 = F.relu(self.bn6(self.enc6(x5))) + t_emb6
        b = F.relu(self.bn_b(self.bottleneck(x6))) + t_emb_b
        # Decoder with skip connections
        d1 = F.relu(self.bn_d1(self.dec1(torch.cat([self.up1(b), x6], dim=1))))
        d2 = F.relu(self.bn_d2(self.dec2(torch.cat([self.up2(d1), x5], dim=1))))
        d3 = F.relu(self.bn_d3(self.dec3(torch.cat([self.up3(d2), x4], dim=1))))
        d4 = F.relu(self.bn_d4(self.dec4(torch.cat([self.up4(d3), x3], dim=1))))
        d5 = F.relu(self.bn_d5(self.dec5(torch.cat([self.up5(d4), x2], dim=1))))
        d6 = F.relu(self.bn_d6(self.dec6(torch.cat([self.up6(d5), x1], dim=1))))
        out = self.out(d6)
        out = self.upsample(out)
        return out

# DDPM
class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000, sampling_steps=50):
        super(DDPM, self).__init__()
        self.model = model
        self.timesteps = timesteps
        self.sampling_steps = sampling_steps
        self.betas = self.linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
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
    def p_sample(self, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape).to(device)
        timesteps = torch.linspace(self.timesteps - 1, 0, steps=self.sampling_steps, dtype=torch.long)
        for i, t_idx in enumerate(timesteps):
            t = torch.full((shape[0],), t_idx, dtype=torch.long, device=device)
            x = self.p_sample(x, t, i)
        return x.clamp(-1, 1)

# Transformations
transform = transforms.Compose([
    transforms.Resize(image_size),   # (H, W)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Main execution
if __name__ == '__main__':
    # Load dataset
    logger.info(f"Loading dataset from {image_dir}...")
    dataset = AstroDataset(image_dir, transform=transform, use_labels=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    logger.info(f"Dataset loaded with {len(dataset)} images")

    # Initialize networks and optimizers
    logger.info("Initializing U-Net and DDPM...")
    unet = UNet(channels=channels).to(device)
    ddpm = DDPM(unet, timesteps=timesteps, sampling_steps=sampling_steps).to(device)
    optimizer = optim.Adam(ddpm.parameters(), lr=lr)
    scaler = GradScaler()  # For mixed precision
    logger.info("Networks and optimizers initialized")

    # Metrics
    fid, inception = setup_metrics(device)

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch}/{num_epochs-1} started")
        for i, real_images in enumerate(dataloader):
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)
            t = torch.randint(0, timesteps, (batch_size_actual,), device=device)
            logger.info(f"Batch {i}/{len(dataloader)}, Actual batch size: {batch_size_actual}, Image shape: {real_images.shape}, Timestep shape: {t.shape}")

            optimizer.zero_grad()
            with autocast('cuda'):
                noise = torch.randn_like(real_images)
                noisy_images = ddpm.q_sample(real_images, t, noise)
                predicted_noise = ddpm(noisy_images, t)
                loss = F.mse_loss(predicted_noise, noise)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if i % 10 == 0:
                logger.info(f'[Epoch {epoch}/{num_epochs-1}] [Batch {i}/{len(dataloader)}] [Loss: {loss.item():.4f}]')

        # Save checkpoints
        checkpoint_path = f"{output_dir}/ddpm_checkpoint_epoch_{epoch}.pth"
        torch.save(ddpm.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Inference and Metrics (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                num_eval_samples = min(2, batch_size_actual)
                fake_images = ddpm.sample((num_eval_samples, channels, *image_size))
                fake_images_display = (fake_images + 1) / 2
                grid = torchvision.utils.make_grid(fake_images_display.cpu(), nrow=1)
                image_path = f"{output_dir}/fake_images_epoch_{epoch+1}.png"
                torchvision.utils.save_image(grid, image_path)
                logger.info(f"Saved generated images to {image_path}")

                fid.reset()
                inception.reset()

                real_images_uint8 = ((real_images[:num_eval_samples] * 0.5 + 0.5) * 255).to(torch.uint8)
                fake_images_uint8 = (fake_images_display[:num_eval_samples] * 255).to(torch.uint8)
                fid.update(real_images_uint8, real=True)
                fid.update(fake_images_uint8, real=False)
                fid_score = fid.compute().item()

                resize_transform = transforms.Resize((299, 299))
                fake_images_resized = resize_transform(fake_images_display[:num_eval_samples])
                inception.update((fake_images_resized * 255).to(torch.uint8))
                inception_score = inception.compute()[0].item()

                save_metrics_and_graphs((fid_score, inception_score), (loss, None), epoch, output_dir, logs_dir)
                logger.info(f"Completed epoch {epoch} with metrics")
        else:
            logger.info(f"Completed epoch {epoch} (metrics skipped)")

    logger.info("Training complete!")