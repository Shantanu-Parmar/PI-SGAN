import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
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
logging.basicConfig(filename=os.path.join(log_dir, f'pisgan_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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

# Hyperparameters
latent_dim = 100
num_classes = 8
image_size = 512
batch_size = 8
lr = 0.00002
n_critic = 5
lambda_gp = 100
lambda_sparse = 0.01
lambda_physics = 0.1

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

# Critic
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

# Physics Loss
def physics_loss(gen_images):
    psf = torch.tensor([
        [0.0025, 0.0054, 0.0113, 0.0235, 0.0113, 0.0054, 0.0025],
        [0.0054, 0.0113, 0.0235, 0.0489, 0.0235, 0.0113, 0.0054],
        [0.0113, 0.0235, 0.0489, 0.1018, 0.0489, 0.0235, 0.0113],
        [0.0235, 0.0489, 0.1018, 0.2119, 0.1018, 0.0489, 0.0235],
        [0.0113, 0.0235, 0.0489, 0.1018, 0.0489, 0.0235, 0.0113],
        [0.0054, 0.0113, 0.0235, 0.0489, 0.0235, 0.0113, 0.0054],
        [0.0025, 0.0054, 0.0113, 0.0235, 0.0113, 0.0054, 0.0025]
    ], device=device).view(1, 1, 7, 7) / 0.9999
    convolved = F.conv2d(gen_images.mean(dim=1, keepdim=True), psf, padding=3)
    psf_loss = torch.mean((gen_images.mean(dim=1, keepdim=True) - convolved) ** 2)
    noise_dev = torch.mean(torch.abs(gen_images))
    return psf_loss + noise_dev

# Dataset
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
        label = torch.randint(0, num_classes, (1,)).item()
        return image, label

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_image_dir = r'D:/Shantanu/MBTR/DATA/Images/train'
dataset = AstroDataset(train_image_dir, transform=transform)

# Train/Val Split
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# Models and optimizers
generator = Generator().to(device)
critic = Critic().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_C = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.999))

# Gradient penalty
def compute_gradient_penalty(critic, real_samples, fake_samples, labels):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    crit_interpolates = critic(interpolates, labels)
    gradients = torch.autograd.grad(outputs=crit_interpolates, inputs=interpolates,
                                   grad_outputs=torch.ones_like(crit_interpolates, device=device),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gp_loss = torch.mean((gradient_norm - 1) ** 2)
    if gp_loss.item() > 1000:
        logging.warning(f"Gradient penalty too large: {gp_loss.item()}")
        return torch.tensor(0.0, device=device)
    return lambda_gp * gp_loss

# Training loop with validation
best_val_loss = float('inf')
for epoch in range(1000):
    # Training phase
    generator.train()
    critic.train()
    train_c_loss, train_g_loss = 0, 0
    for i, (real_images, labels) in enumerate(train_loader):
        real_images = real_images.to(device)
        labels = torch.tensor(labels, dtype=torch.long).detach().clone().to(device)
        current_batch_size = real_images.size(0)

        # Train Critic
        for _ in range(n_critic):
            optimizer_C.zero_grad()
            crit_real = critic(real_images, labels).mean()
            noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            fake_labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
            fake_images = generator(noise, fake_labels)
            crit_fake = critic(fake_images.detach(), fake_labels).mean()
            gp = compute_gradient_penalty(critic, real_images, fake_images.detach(), labels)
            c_loss = -crit_real + crit_fake + gp
            c_loss.backward()
            if torch.isnan(c_loss).any() or torch.isinf(c_loss).any():
                logging.warning(f"NaN/Inf in C loss at epoch {epoch}, batch {i}")
                c_loss = torch.tensor(0.0, device=device)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            optimizer_C.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_images = generator(noise, fake_labels)
        crit_fake = critic(fake_images, fake_labels).mean()
        sparse_loss = lambda_sparse * torch.mean(torch.abs(fake_images))
        physics_loss_val = lambda_physics * physics_loss(fake_images)
        g_loss = -crit_fake + sparse_loss + physics_loss_val
        g_loss.backward()
        if torch.isnan(g_loss).any() or torch.isinf(g_loss).any():
            logging.warning(f"NaN/Inf in G loss at epoch {epoch}, batch {i}")
            g_loss = torch.tensor(0.0, device=device)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer_G.step()

        train_c_loss += c_loss.item()
        train_g_loss += g_loss.item()

        if i % 100 == 0 and i > 0:
            avg_c_loss = train_c_loss / (i + 1)
            avg_g_loss = train_g_loss / (i + 1)
            print(f'[Epoch {epoch}/{1000}] [Batch {i}/{len(train_loader)}] [C loss: {avg_c_loss:.4f}] [G loss: {avg_g_loss:.4f}]')
            logging.info(f'Train - Epoch {epoch}, Batch {i}, C loss: {avg_c_loss:.4f}, G loss: {avg_g_loss:.4f}')

        # Validation phase
        generator.eval()
        critic.eval()
        val_c_loss, val_g_loss = 0, 0
        for i, (real_images, labels) in enumerate(val_loader):
            real_images = real_images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).detach().clone().to(device)
            current_batch_size = real_images.size(0)

            noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            fake_labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
            with torch.no_grad():
                fake_images = generator(noise, fake_labels)
                crit_real = critic(real_images, labels).mean()
                crit_fake = critic(fake_images, fake_labels).mean()

            # Compute gradient penalty in a gradient-enabled context
            real_images_gp = real_images.detach().requires_grad_(True)
            fake_images_gp = fake_images.detach().requires_grad_(True)
            alpha = torch.rand(real_images_gp.size(0), 1, 1, 1, device=device).expand_as(real_images_gp)
            interpolates = (alpha * real_images_gp + (1 - alpha) * fake_images_gp)
            #print(f"Interpolates requires grad: {interpolates.requires_grad}")  # Debug
            crit_interpolates = critic(interpolates, labels)
            gradients = torch.autograd.grad(outputs=crit_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(crit_interpolates, device=device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
            gp_loss = torch.mean((gradient_norm - 1) ** 2)
            if gp_loss.item() > 1000:
                logging.warning(f"Gradient penalty too large: {gp_loss.item()}")
                gp = torch.tensor(0.0, device=device)
            else:
                gp = lambda_gp * gp_loss

            c_loss = -crit_real + crit_fake + gp
            with torch.no_grad():
                crit_fake_val = critic(fake_images, fake_labels).mean()
                sparse_loss_val = lambda_sparse * torch.mean(torch.abs(fake_images))
                physics_loss_val = lambda_physics * physics_loss(fake_images)
                g_loss = -crit_fake_val + sparse_loss_val + physics_loss_val

            val_c_loss += c_loss.item()
            val_g_loss += g_loss.item()

        avg_val_c_loss = val_c_loss / len(val_loader)
        avg_val_g_loss = val_g_loss / len(val_loader)
        print(f'[Epoch {epoch}/{1000}] [Validation] [C loss: {avg_val_c_loss:.4f}] [G loss: {avg_val_g_loss:.4f}]')
        logging.info(f'Validation - Epoch {epoch}, C loss: {avg_val_c_loss:.4f}, G loss: {avg_val_g_loss:.4f}')

        if avg_val_g_loss < best_val_loss:
            best_val_loss = avg_val_g_loss
            torch.save(generator.state_dict(), os.path.join(log_dir, f'generator_best.pth'))
            torch.save(critic.state_dict(), os.path.join(log_dir, f'critic_best.pth'))
            logging.info(f'Saved best model at epoch {epoch} with G loss: {best_val_loss:.4f}')

    # Save checkpoint
    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_C_state_dict': optimizer_C.state_dict(),
            'loss': {'c_loss': train_c_loss, 'g_loss': train_g_loss}
        }, os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        logging.info(f'Saved checkpoint at epoch {epoch+1}')

    # Save samples
    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            noise = torch.randn(16, latent_dim, 1, 1, device=device)
            fixed_labels = torch.full((16,), 0, dtype=torch.long, device=device)
            fake_images = generator(noise, fixed_labels)
            fake_images = (fake_images + 1) / 2
            torchvision.utils.save_image(fake_images, os.path.join(log_dir, f'pisgan_sample_epoch_{epoch+1}_class0.png'))

# Inference mode
def inference(generator, num_samples=16, class_id=0, save_path=r'D:/Shantanu/MBTR/DATA/inference'):
    os.makedirs(save_path, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, 1, 1, device=device)
        labels = torch.full((num_samples,), class_id, dtype=torch.long, device=device)
        fake_images = generator(noise, labels)
        fake_images = (fake_images + 1) / 2
        torchvision.utils.save_image(fake_images, os.path.join(save_path, f'inference_samples_class{class_id}.png'))
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(fake_images[i].cpu().permute(1, 2, 0))
            ax.axis('off')
        plt.savefig(os.path.join(save_path, f'inference_vis_class{class_id}.png'))
        plt.close()

# Run inference after training
inference(generator)
print("Training and inference complete!")
logging.info("Training and inference complete.")