import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import matplotlib.pyplot as plt
import logging
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import torch.utils.checkpoint as checkpoint

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset-specific configuration
dataset_name = "MobilTelesco"
image_dir = f"F:/Shantanu/PISGAN/DATA/Images/{dataset_name}/train"
output_dir = f"outputs/PI-SGAN_{dataset_name}"
logs_dir = f"logs/PI-SGAN_{dataset_name}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{logs_dir}/pi_sgan_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Hyperparameters
image_size = (1024, 1365)  # New target resolution (H, W) maintaining 3:4 aspect ratio
channels = 3
batch_size = 1  # Fits 8GB VRAM
latent_dim = 100
num_epochs = 100
lr = 0.0002
lambda_physics = 0.1  # Weight for physics loss
gamma = 10.0  # Gradient penalty coefficient
val_split = 0.2  # 20% validation split

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
        return None  # Unconditional for MobilTelesco

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Metrics setup
def setup_metrics(device):
    logger.info("Initializing metrics...")
    fid = FrechetInceptionDistance().to(device)
    inception = InceptionScore().to(device)
    logger.info("Metrics initialized: FID, Inception")
    return fid, inception

# Save metrics and graphs
def save_metrics_and_graphs(metrics, losses, epoch, split, output_dir, logs_dir):
    fid_score, inception_score = metrics
    loss_g, loss_d = losses
    logger.info(f"Epoch {epoch}, {split} - FID: {fid_score:.4f}, Inception Score: {inception_score:.4f}, G Loss: {loss_g:.4f}, D Loss: {loss_d:.4f}")
    with open(f"{logs_dir}/metrics.log", "a") as f:
        f.write(f"Epoch {epoch}, {split}, FID: {fid_score:.4f}, Inception Score: {inception_score:.4f}, G Loss: {loss_g:.4f}, D Loss: {loss_d:.4f}\n")
    plt.figure()
    plt.plot(range(epoch + 1), [loss_g.item() for _ in range(epoch + 1)], label="Generator Loss")
    plt.plot(range(epoch + 1), [loss_d.item() for _ in range(epoch + 1)], label="Discriminator Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_graph_{split}_epoch_{epoch+1}.png")
    plt.close()
    logger.info(f"Saved loss graph to {output_dir}/loss_graph_{split}_epoch_{epoch+1}.png")

# Physics Loss
def compute_physics_loss(images, snr_target=10.0):
    snr = torch.mean(images, dim=[1, 2, 3]) / (torch.std(images, dim=[1, 2, 3]) + 1e-6)
    snr_loss = F.mse_loss(snr, torch.tensor(snr_target, device=images.device).repeat(images.size(0)))
    noise = torch.abs(images - torch.mean(images, dim=[1, 2, 3], keepdim=True))
    poisson_loss = torch.mean(noise - torch.log(noise + 1e-6))
    return snr_loss + 0.1 * poisson_loss



class Generator(nn.Module):
    def __init__(self, channels=3, latent_dim=100, time_dim=256):
        super(Generator, self).__init__()
        self.time_dim = time_dim
        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        ).to(device)
        self.time_proj1 = nn.Linear(time_dim, 256).to(device)
        
        # Initial layer for 1024x1365, starting at 32x43
        self.initial = nn.Linear(latent_dim, 512 * 32 * 43).to(device)
        self.enc1 = nn.Conv2d(512, 256, 4, 2, 1, bias=False).to(device)  # 32x43 -> 16x22
        self.bn1 = nn.BatchNorm2d(256).to(device)
        self.bottleneck = nn.Conv2d(256, 256, 3, 1, 1, bias=False).to(device)  # 16x22 -> 16x22 (reduced from 512)
        self.bn_b = nn.BatchNorm2d(256).to(device)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1, output_padding=1, bias=False).to(device)  # 16x22 -> 32x43
        self.dec1 = nn.Conv2d(384, 128, 3, 1, 1, bias=False).to(device)  # 128 + 256
        self.bn_d1 = nn.BatchNorm2d(128).to(device)
        self.up2 = nn.ConvTranspose2d(128, 128, 4, 2, 1, output_padding=1, bias=False).to(device)  # 32x43 -> 64x86
        self.dec2 = nn.Conv2d(256, 128, 3, 1, 1, bias=False).to(device)  # 128 + 128
        self.bn_d2 = nn.BatchNorm2d(128).to(device)
        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=1, bias=False).to(device)  # 64x86 -> 128x172
        self.dec3 = nn.Conv2d(192, 64, 3, 1, 1, bias=False).to(device)  # 64 + 128
        self.bn_d3 = nn.BatchNorm2d(64).to(device)
        self.up4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, output_padding=1, bias=False).to(device)  # 128x172 -> 256x344
        self.dec4 = nn.Conv2d(96, 32, 3, 1, 1, bias=False).to(device)  # 64 + 32
        self.bn_d4 = nn.BatchNorm2d(32).to(device)
        self.up5 = nn.ConvTranspose2d(32, 16, 4, 2, 1, output_padding=1, bias=False).to(device)  # 256x344 -> 512x688
        self.dec5 = nn.Conv2d(48, 16, 3, 1, 1, bias=False).to(device)  # 32 + 16
        self.bn_d5 = nn.BatchNorm2d(16).to(device)
        self.dec6 = nn.Conv2d(24, 8, 3, 1, 1, bias=False).to(device)  # 16 + 8
        self.bn_d6 = nn.BatchNorm2d(8).to(device)
        self.dec7 = nn.Conv2d(8, channels, 3, 1, 1, bias=False).to(device)  # Final to 3 channels
        self.upsample = nn.Upsample(size=image_size, mode='bilinear', align_corners=False).to(device)
        self.upsample_skip1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).to(device)
        self.upsample_skip2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).to(device)
        self.upsample_skip3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).to(device)
        self.upsample_skip4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).to(device)
        self.upsample_skip5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).to(device)
        self.conv_reduce = nn.Conv2d(256, 128, 1, 1, 0, bias=False).to(device)  # Channel reduction

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq.to(t.device))
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq.to(t.device))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def _forward_block(self, x, upsample, dec, bn, input_tensor=None):
        if input_tensor is not None:
            # Ensure dimensions match by cropping or padding if necessary
            _, _, h, w = input_tensor.size()
            if x.size(2) != h or x.size(3) != w:
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            x = torch.cat([x, input_tensor], dim=1)
        x = dec(x)
        x = bn(x)
        x = F.relu(x)
        return x

    def forward(self, z, t=None):
        if t is not None:
            t_emb = self.pos_encoding(t, self.time_dim)
            t_emb = self.time_emb(t_emb)
            t_emb1 = self.time_proj1(t_emb)[:, :, None, None]
        x = self.initial(z).view(-1, 512, 32, 43)
        x1 = F.relu(self.bn1(self.enc1(x)) + (t_emb1 if t is not None else 0))
        b = F.relu(self.bn_b(self.bottleneck(x1)))
        x1_upsampled1 = self.upsample_skip1(x1)
        d1 = checkpoint.checkpoint(self._forward_block, self.up1(b), None, self.dec1, self.bn_d1, x1_upsampled1, use_reentrant=False)
        if 't_emb1' in locals():
            del t_emb1
        del x1_upsampled1
        x1_upsampled1 = self.upsample_skip1(x1)
        x1_upsampled2 = self.upsample_skip2(x1_upsampled1)
        x1_reduced = self.conv_reduce(x1_upsampled2)
        d2 = checkpoint.checkpoint(self._forward_block, self.up2(d1), None, self.dec2, self.bn_d2, x1_reduced, use_reentrant=False)
        del x1_upsampled1, x1_upsampled2, x1_reduced, d1
        x_upsampled1 = self.upsample_skip1(x)
        x_upsampled2 = self.upsample_skip2(x_upsampled1)
        d3 = checkpoint.checkpoint(self._forward_block, self.up3(d2), None, self.dec3, self.bn_d3, x_upsampled2, use_reentrant=False)
        del x, x_upsampled1, d2
        x_upsampled3 = self.upsample_skip3(x_upsampled2)
        d4 = checkpoint.checkpoint(self._forward_block, self.up4(d3), None, self.dec4, self.bn_d4, x_upsampled3, use_reentrant=False)
        del x_upsampled2, d3
        x_upsampled4 = self.upsample_skip4(x_upsampled3)
        d5 = checkpoint.checkpoint(self._forward_block, self.up5(d4), None, self.dec5, self.bn_d5, x_upsampled4, use_reentrant=False)
        del x_upsampled3, d4
        d6 = checkpoint.checkpoint(self._forward_block, d5, None, self.dec6, self.bn_d6, x_upsampled4, use_reentrant=False)
        del d5, x_upsampled4
        d7 = torch.tanh(self.dec7(d6))
        del d6
        del b
        del x1
        return self.upsample(d7)



# Discriminator (Stacked with Physics-Informed Head)
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, 4, 2, 1, bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(64).to(device)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False).to(device)
        self.bn2 = nn.BatchNorm2d(128).to(device)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False).to(device)
        self.bn3 = nn.BatchNorm2d(256).to(device)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False).to(device)
        self.bn4 = nn.BatchNorm2d(512).to(device)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0, bias=False).to(device)
        self.spectral_norm = nn.utils.spectral_norm

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        return self.conv5(x).view(-1)

def train_pi_sgan(generator, discriminator, train_loader, val_loader, num_epochs, lr, lambda_physics, gamma, accumulation_steps=32):
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    scaler_g = GradScaler()
    scaler_d = GradScaler()
    fid_train, inception_train = setup_metrics(device)
    fid_val, inception_val = setup_metrics(device)
    best_fid = float('inf')
    checkpoint_path_best = f"{output_dir}/pi_sgan_best_checkpoint.pth"

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        logger.info(f"Epoch {epoch}/{num_epochs-1} - Training started")
        g_loss_accum = 0
        d_loss_accum = 0
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for i, real_images in enumerate(train_loader):
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_images = generator(z)

            # Discriminator
            with autocast('cuda'):
                real_score = discriminator(real_images)
                fake_score = discriminator(fake_images.detach())
                physics_loss_d = compute_physics_loss(real_images)
                d_loss = -torch.mean(real_score) + torch.mean(fake_score) + gamma * physics_loss_d
                gp = gradient_penalty(discriminator, real_images, fake_images)
                d_loss += gamma * gp
            scaler_d.scale(d_loss / accumulation_steps).backward()
            d_loss_accum += d_loss.item()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler_d.step(optimizer_d)
                scaler_d.update()
                optimizer_d.zero_grad()

            # Generator
            with autocast('cuda'):
                fake_score = discriminator(fake_images)
                physics_loss_g = compute_physics_loss(fake_images)
                g_loss = -torch.mean(fake_score) + lambda_physics * physics_loss_g
            scaler_g.scale(g_loss / accumulation_steps).backward()
            g_loss_accum += g_loss.item()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()
                optimizer_g.zero_grad()

            if i % (10 * accumulation_steps) == 0:
                logger.info(f'[Epoch {epoch}/{num_epochs-1}] [Batch {i}/{len(train_loader)}] [G Loss: {g_loss_accum:.4f}, D Loss: {d_loss_accum:.4f}]')
                g_loss_accum = 0
                d_loss_accum = 0

        # Validation (limit to one image)
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            for i, val_images in enumerate(val_loader):
                batch_size_actual = val_images.size(0)
                val_images = val_images.to(device)
                z_val = torch.randn(batch_size_actual, latent_dim, device=device)
                fake_val_images = generator(z_val)
                fid_val.update(((val_images[0:1] * 0.5 + 0.5) * 255).to(torch.uint8), real=True)
                fid_val.update(((fake_val_images[0:1] * 0.5 + 0.5) * 255).to(torch.uint8), real=False)
                resize_transform = transforms.Resize((299, 299))
                inception_val.update((resize_transform((fake_val_images[0:1] * 0.5 + 0.5) * 255).to(torch.uint8)))
                break

        fid_val_score = fid_val.compute().item()
        inception_val_score = inception_val.compute()[0].item()
        save_metrics_and_graphs((fid_val_score, inception_val_score), (g_loss.item(), d_loss.item()), epoch, "Validation", output_dir, logs_dir)

        # Training Metrics (limit to one image)
        with torch.no_grad():
            for i, train_images in enumerate(train_loader):
                batch_size_actual = train_images.size(0)
                train_images = train_images.to(device)
                z_train = torch.randn(batch_size_actual, latent_dim, device=device)
                fake_train_images = generator(z_train)
                fid_train.update(((train_images[0:1] * 0.5 + 0.5) * 255).to(torch.uint8), real=True)
                fid_train.update(((fake_train_images[0:1] * 0.5 + 0.5) * 255).to(torch.uint8), real=False)
                resize_transform = transforms.Resize((299, 299))
                inception_train.update((resize_transform((fake_train_images[0:1] * 0.5 + 0.5) * 255).to(torch.uint8)))
                break

        fid_train_score = fid_train.compute().item()
        inception_train_score = inception_train.compute()[0].item()
        save_metrics_and_graphs((fid_train_score, inception_train_score), (g_loss.item(), d_loss.item()), epoch, "Training", output_dir, logs_dir)

        # Checkpointing
        checkpoint_path = f"{output_dir}/pi_sgan_checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'fid_val': fid_val_score,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        if fid_val_score < best_fid:
            best_fid = fid_val_score
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'fid_val': fid_val_score,
            }, f"{output_dir}/pi_sgan_best_checkpoint.pth")
            logger.info(f"Saved best checkpoint with FID: {best_fid:.4f}")

    logger.info("Training complete!")

# Gradient Penalty
def gradient_penalty(discriminator, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates = interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                   grad_outputs=torch.ones_like(d_interpolates),
                                   create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# Inference Function
def infer_pi_sgan(checkpoint_path, num_samples=2):
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    generator = Generator(channels=channels, latent_dim=latent_dim).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        fake_images = generator(z)
        fake_images_display = (fake_images + 1) / 2
        grid = torchvision.utils.make_grid(fake_images_display.cpu(), nrow=1)
        image_path = f"{output_dir}/inferred_images.png"
        torchvision.utils.save_image(grid, image_path)
        logger.info(f"Saved inferred images to {image_path}")
    return fake_images_display

# Transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Main execution
if __name__ == '__main__':
    # Load dataset and create train/val split
    logger.info(f"Loading dataset from {image_dir}...")
    dataset = AstroDataset(image_dir, transform=transform, use_labels=False)
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True, drop_last=True)
    logger.info(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val")

    # Initialize networks
    logger.info("Initializing Generator and Discriminator...")
    generator = Generator(channels=channels, latent_dim=latent_dim).to(device)
    discriminator = Discriminator(channels=channels).to(device)
    logger.info("Networks initialized")

    # Train
    train_pi_sgan(generator, discriminator, train_loader, val_loader, num_epochs, lr, lambda_physics, gamma, accumulation_steps=32)
    # Example Inference
    checkpoint_path = f"{output_dir}/pi_sgan_best_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        infer_pi_sgan(checkpoint_path)
    else:
        logger.warning(f"Best checkpoint not found at {checkpoint_path}")