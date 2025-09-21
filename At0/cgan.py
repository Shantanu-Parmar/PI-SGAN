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
num_classes = 8  # Your 8 classes (Pleiades, Jupiter, etc.)
image_size = 64
batch_size = 32
num_epochs = 100  # Increased for better convergence
lr = 0.0002
beta1 = 0.5
label_smooth = 0.9  # For discriminator real labels

# Generator Network (Conditional)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)  # Embed class label
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 512, 4, 1, 0, bias=False),  # Concat noise + label
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

# Discriminator Network (Conditional)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, image_size * image_size)  # Embed label to image size
        self.main = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),  # Input channels: 3 (image) + 1 (label map)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_emb = self.label_emb(labels).view(-1, 1, image_size, image_size)
        disc_input = torch.cat((x, label_emb), dim=1)
        return self.main(disc_input).view(-1, 1)

# Custom Dataset (Add random labels for prototype; adapt to your .txt annotations)
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
        label = torch.randint(0, num_classes, (1,)).item()  # Random class; replace with real from .txt
        return image, label

# Transformations (Normalize for dark astro images)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.1, 0.1, 0.1), (0.3, 0.3, 0.3))  # Adjusted for low-light data
])

# Load dataset
train_image_dir = r'D:/Shantanu/MBTR/DATA/Images/train'
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
    for i, (real_images, labels) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        labels = labels.to(device)
        
        # Labels with smoothing
        real_label = torch.full((batch_size, 1), label_smooth, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()
        output_real = discriminator(real_images, labels)
        d_loss_real = criterion(output_real, real_label)
        
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)  # Random fake labels
        fake_images = generator(noise, fake_labels)
        output_fake = discriminator(fake_images.detach(), fake_labels)
        d_loss_fake = criterion(output_fake, fake_label)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        output_fake = discriminator(fake_images, fake_labels)
        g_loss = criterion(output_fake, real_label)
        g_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f'[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]')

    # Save generated images (conditioned on class 0 for Pleiades example)
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            noise = torch.randn(16, latent_dim, 1, 1).to(device)
            fixed_labels = torch.full((16,), 0, dtype=torch.long).to(device)  # Condition on class 0
            fake_images = generator(noise, fixed_labels)
            fake_images = (fake_images + 1) / 2  # Unnormalize
            grid = torchvision.utils.make_grid(fake_images.cpu(), nrow=4)
            torchvision.utils.save_image(grid, f'fake_images_epoch_{epoch+1}_class0.png')

print("Training complete!")