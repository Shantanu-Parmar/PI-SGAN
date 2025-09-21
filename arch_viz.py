import torch
import torch.nn as nn
from torchviz import make_dot

# Define Generator
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        
        self.attention = nn.Sequential(nn.Conv2d(256, 1, 1), nn.Sigmoid())
        
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, out_channels, 4, 2, 1), nn.Tanh())
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        att = self.attention(d3)
        d3 = d3 * att
        
        u1 = self.up1(d3)
        u2 = self.up2(u1 + d2)
        u3 = self.up3(u2 + d1)
        return u3

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model instances
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Dummy input for visualization (match expected input shape, e.g., 3x200x200)
dummy_input = torch.randn(1, 3, 200, 200).to(device)

# Visualize Generator
generator_output = generator(dummy_input)
dot_generator = make_dot(generator_output, params=dict(generator.named_parameters()))
dot_generator.format = 'png'
dot_generator.render("generator_architecture")

# Visualize Discriminator
discriminator_output = discriminator(dummy_input)
dot_discriminator = make_dot(discriminator_output, params=dict(discriminator.named_parameters()))
dot_discriminator.format = 'png'
dot_discriminator.render("discriminator_architecture")

print("Architecture visualizations saved as generator_architecture.png and discriminator_architecture.png")