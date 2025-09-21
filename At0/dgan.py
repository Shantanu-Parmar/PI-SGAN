import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
num_classes = 8
image_size = 64
batch_size = 64  # Initial batch size for DataLoader
num_epochs = 200
lr = 0.0001  # Lowered for WGAN
beta1 = 0.5
n_critic = 5  # Train critic more often
lambda_gp = 10  # Gradient penalty weight

# Generator (Conditional)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 512, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels).view(-1, latent_dim, 1, 1)
        gen_input = torch.cat((noise, label_emb), dim=1)
        return self.main(gen_input)

# Critic (Conditional, no BN for stability)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.label_emb = nn.Embedding(num_classes, image_size * image_size)
        self.main = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x, labels):
        label_emb = self.label_emb(labels).view(-1, 1, image_size, image_size)
        crit_input = torch.cat((x, label_emb), dim=1)
        return self.main(crit_input).view(-1)

# Compute gradient penalty
def compute_gradient_penalty(critic, real_samples, fake_samples, labels):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    crit_interpolates = critic(interpolates, labels)
    gradients = torch.autograd.grad(outputs=crit_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(crit_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Dataset (random labels; adapt to .txt)
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

# Transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.1, 0.1, 0.1), (0.3, 0.3, 0.3))  # Low-light adjusted
])

# Load dataset
train_image_dir = r'D:\Shantanu\MBTR\DATA\Images\train'
dataset = AstroDataset(train_image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize
generator = Generator().to(device)
critic = Critic().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        real_images = real_images.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)  # Use long for embeddings
        
        current_batch_size = real_images.size(0)  # Dynamic batch size

        # Train Critic
        for _ in range(n_critic):
            optimizer_C.zero_grad()
            crit_real = critic(real_images, labels).mean()
            
            noise = torch.randn(current_batch_size, latent_dim, 1, 1).to(device)
            fake_labels = torch.randint(0, num_classes, (current_batch_size,)).to(device)
            fake_images = generator(noise, fake_labels)
            crit_fake = critic(fake_images.detach(), fake_labels).mean()
            
            gp = lambda_gp * compute_gradient_penalty(critic, real_images, fake_images.detach(), labels)
            c_loss = -crit_real + crit_fake + gp
            c_loss.backward()
            optimizer_C.step()

        # Train Generator
        optimizer_G.zero_grad()
        noise = torch.randn(current_batch_size, latent_dim, 1, 1).to(device)
        fake_labels = torch.randint(0, num_classes, (current_batch_size,)).to(device)
        fake_images = generator(noise, fake_labels)
        g_loss = -critic(fake_images, fake_labels).mean()
        g_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f'[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [C loss: {c_loss.item():.4f}] [G loss: {g_loss.item():.4f}]')

    # Save samples
    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            noise = torch.randn(16, latent_dim, 1, 1).to(device)
            fixed_labels = torch.full((16,), 0, dtype=torch.long).to(device)
            fake_images = generator(noise, fixed_labels)
            fake_images = (fake_images + 1) / 2
            grid = torchvision.utils.make_grid(fake_images.cpu(), nrow=4)
            torchvision.utils.save_image(grid, f'fake_images_epoch_{epoch+1}_class0.png')

print("Training complete!")