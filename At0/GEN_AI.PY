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
image_size = 64
batch_size = 32
num_epochs = 50
lr = 0.0002
beta1 = 0.5

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: (batch_size, 100, 1, 1)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: (batch_size, 512, 4, 4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: (batch_size, 256, 8, 8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: (batch_size, 128, 16, 16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: (batch_size, 64, 32, 32)
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output: (batch_size, 3, 64, 64)
        )

    def forward(self, x):
        return self.main(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (batch_size, 3, 64, 64)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, 64, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, 128, 16, 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, 256, 8, 8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, 512, 4, 4)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output: (batch_size, 1, 1, 1)
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)

# Custom Dataset for MobilTelesco
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
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] for tanh output
])

# Load dataset
train_image_dir = r'D:/Shantanu/MBTR/DATA/Images/train'  # Path to your train images
dataset = AstroDataset(train_image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize networks and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, real_images in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Labels
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        output_real = discriminator(real_images)
        d_loss_real = criterion(output_real, real_label)
        
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(output_fake, fake_label)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        output_fake = discriminator(fake_images)
        g_loss = criterion(output_fake, real_label)
        g_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f'[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]')

    # Save generated images
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            fake_images = generator(torch.randn(16, latent_dim, 1, 1).to(device))
            fake_images = (fake_images + 1) / 2  # Unnormalize to [0, 1]
            grid = torchvision.utils.make_grid(fake_images.cpu(), nrow=4)
            torchvision.utils.save_image(grid, f'fake_images_epoch_{epoch+1}.png')

print("Training complete!")