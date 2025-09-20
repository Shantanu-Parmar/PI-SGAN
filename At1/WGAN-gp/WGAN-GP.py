import torch
import torch.nn as nn
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
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset-specific configuration
dataset_name = "MobilTelesco"
image_dir = f"D:/Shantanu/MBTR/DATA/Images/{dataset_name}/train"
output_dir = f"outputs/WGAN-GP_{dataset_name}"
logs_dir = f"logs/WGAN-GP_{dataset_name}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{logs_dir}/wgangan_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Hyperparameters
latent_dim = 100
image_size = (1536, 2048)  # Target resolution (H, W)
batch_size = 2  # Fits 8GB VRAM
num_epochs = 100
lr = 0.0002
beta1 = 0.5
n_critic = 5
lambda_gp = 10

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
        # Placeholder: Not used for WGAN-GP
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
    c_loss, g_loss = losses
    logger.info(f"Epoch {epoch}, FID: {fid_score:.4f}, Inception Score: {inception_score:.4f}")
    with open(f"{logs_dir}/metrics.log", "a") as f:
        f.write(f"Epoch {epoch}, FID: {fid_score:.4f}, Inception Score: {inception_score:.4f}\n")
    plt.figure()
    plt.plot(range(epoch + 1), [c_loss.item() for _ in range(epoch + 1)], label="C Loss")
    plt.plot(range(epoch + 1), [g_loss.item() for _ in range(epoch + 1)], label="G Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_graph_epoch_{epoch+1}.png")
    plt.close()
    logger.info(f"Saved loss graph to {output_dir}/loss_graph_epoch_{epoch+1}.png")

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_size=(1536, 2048)):
        super(Generator, self).__init__()
        self.base = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 8, 1, 0, bias=False),  # 1 -> 8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 8 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),    # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),    # 128 -> 256
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),     # 256 -> 512
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False),      # 512 -> 1024
            nn.Tanh()
        )
        self.upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        out = self.base(x)
        out = self.upsample(out)
        return out

# Critic
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),   # 1536 -> 768
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 768 -> 384
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # 384 -> 192
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),# 192 -> 96
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),# 96 -> 48
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),# 48 -> 24
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)

# Gradient Penalty
def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    crit_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(outputs=crit_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(crit_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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
    logger.info("Initializing generator and critic...")
    generator = Generator(latent_dim, out_size=image_size).to(device)
    critic = Critic().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, 0.999))
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
            logger.info(f"Batch {i}/{len(dataloader)}, Actual batch size: {batch_size_actual}, Image shape: {real_images.shape}")

            # Train Critic
            for _ in range(n_critic):
                optimizer_C.zero_grad()
                with autocast():
                    crit_real = critic(real_images).mean()
                    noise = torch.randn(batch_size_actual, latent_dim, 1, 1).to(device)
                    fake_images = generator(noise)
                    crit_fake = critic(fake_images.detach()).mean()
                    gp = lambda_gp * compute_gradient_penalty(critic, real_images, fake_images.detach())
                    c_loss = -crit_real + crit_fake + gp
                
                scaler.scale(c_loss).backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                scaler.step(optimizer_C)
                scaler.update()

            # Train Generator
            optimizer_G.zero_grad()
            with autocast():
                fake_images = generator(noise)
                g_loss = -critic(fake_images).mean()
            
            scaler.scale(g_loss).backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler.step(optimizer_G)
            scaler.update()

            if i % 10 == 0:
                logger.info(f'[Epoch {epoch}/{num_epochs-1}] [Batch {i}/{len(dataloader)}] [C loss: {c_loss.item():.4f}] [G loss: {g_loss.item():.4f}]')

        # Save checkpoints
        checkpoint_g_path = f"{output_dir}/generator_checkpoint_epoch_{epoch}.pth"
        checkpoint_c_path = f"{output_dir}/critic_checkpoint_epoch_{epoch}.pth"
        torch.save(generator.state_dict(), checkpoint_g_path)
        torch.save(critic.state_dict(), checkpoint_c_path)
        logger.info(f"Saved checkpoints: {checkpoint_g_path}, {checkpoint_c_path}")

        # Inference and Metrics (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                num_eval_samples = min(2, batch_size_actual)
                fake_images = generator(torch.randn(num_eval_samples, latent_dim, 1, 1).to(device))
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

                save_metrics_and_graphs((fid_score, inception_score), (c_loss, g_loss), epoch, output_dir, logs_dir)
                logger.info(f"Completed epoch {epoch} with metrics")
        else:
            logger.info(f"Completed epoch {epoch} (metrics skipped)")

    logger.info("Training complete!")