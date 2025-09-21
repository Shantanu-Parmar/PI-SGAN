import torch
import torch.nn as nn
import hiddenlayer as hl

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

# Define color map for layer types
color_map = {
    'Conv2d': 'orange',
    'BatchNorm2d': 'lightblue',
    'LeakyReLU': 'green',
    'ReLU': 'green',
    'Sigmoid': 'pink',
    'Tanh': 'pink',
    'ConvTranspose2d': 'yellow'
}

# Visualize Generator
graph_generator = hl.build_graph(generator, torch.zeros([1, 3, 200, 200]).to(device))
graph_generator.theme = hl.graph.THEMES["blue"].copy()
graph_generator.theme["fill"] = color_map
graph_generator.save("generator_architecture.png", format="png")

# Visualize Discriminator
graph_discriminator = hl.build_graph(discriminator, torch.zeros([1, 3, 200, 200]).to(device))
graph_discriminator.theme = hl.graph.THEMES["blue"].copy()
graph_discriminator.theme["fill"] = color_map
graph_discriminator.save("discriminator_architecture.png", format="png")

print("Architecture visualizations saved as generator_architecture.png and discriminator_architecture.png")