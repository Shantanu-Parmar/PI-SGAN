import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.utils as vutils

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()])
logger = logging.getLogger()

class AstroDataset(Dataset):
    def __init__(self, mobil_dir, ref_dir, split, transform=None):
        self.mobil_dir = os.path.join(mobil_dir, split)
        self.ref_dir = ref_dir  
        self.transform = transform
        self.mobil_images = []
        self.ref_images = []

        if os.path.exists(self.mobil_dir):
            for obj in os.listdir(self.mobil_dir):
                mobil_obj_dir = os.path.join(self.mobil_dir, obj)
                if os.path.isdir(mobil_obj_dir):
                    self.mobil_images.extend([
                        os.path.join(mobil_obj_dir, x)
                        for x in os.listdir(mobil_obj_dir)
                        if x.lower().endswith(".jpg")
                    ])
        else:
            logger.warning(f"Mobil directory {self.mobil_dir} does not exist")

        if os.path.exists(self.ref_dir):
            for obj in os.listdir(self.ref_dir):
                ref_obj_dir = os.path.join(self.ref_dir, obj)
                if os.path.isdir(ref_obj_dir):
                    self.ref_images.extend([
                        os.path.join(ref_obj_dir, x)
                        for x in os.listdir(ref_obj_dir)
                        if x.lower().endswith(".png")
                    ])
        else:
            logger.warning(f"Reference directory {self.ref_dir} does not exist")

        if not self.mobil_images:
            logger.warning(f"No Mobil images found in {self.mobil_dir}")
        if not self.ref_images:
            logger.warning(f"No Reference images found in {self.ref_dir}")

    def __len__(self):
        return min(len(self.mobil_images), len(self.ref_images))

    def __getitem__(self, idx):
        mobil_img = Image.open(self.mobil_images[idx % len(self.mobil_images)]).convert("RGB")
        ref_img = Image.open(self.ref_images[idx % len(self.ref_images)]).convert("RGB")

        if self.transform:
            mobil_img = self.transform(mobil_img)
            ref_img = self.transform(ref_img)

        return mobil_img, ref_img
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        # Adjusted for 800x800 input: Downsample to 100x100, then upsample back
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2))  # 800 -> 400
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))  # 400 -> 200
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))  # 200 -> 100
        self.down4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))  # 100 -> 50
        
        self.attention = nn.Sequential(nn.Conv2d(512, 1, 1), nn.Sigmoid())
        
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())  # 50 -> 100
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())  # 100 -> 200
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())  # 200 -> 400
        self.up4 = nn.Sequential(nn.ConvTranspose2d(64, out_channels, 4, 2, 1), nn.Tanh())  # 400 -> 800
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        att = self.attention(d4)
        d4 = d4 * att
        
        u1 = self.up1(d4)
        u2 = self.up2(u1 + d3)
        u3 = self.up3(u2 + d2)
        u4 = self.up4(u3 + d1)
        return u4

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),  # 800 -> 400
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 400 -> 200
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),  # 200 -> 100
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),  # 100 -> 50
            nn.Conv2d(512, 1, 4, 1, 0), nn.Sigmoid()  # 50 -> 47 (adjusted for output)
        )
    
    def forward(self, x):
        return self.model(x)

class CMRA_GAN:
    def __init__(self, mobil_dir, ref_dir):
        self.mobil_dir = mobil_dir
        self.ref_dir = ref_dir
        self.generators = [Generator().to(device) for _ in range(3)]
        self.discriminators = [Discriminator().to(device) for _ in range(2)]
        self.optimizers_g = [optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999)) for g in self.generators]
        self.optimizers_d = [optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999)) for d in self.discriminators]
        self.criterion = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_step(self, mobil_batch, ref_batch):
        batch_size = mobil_batch.size(0)
        real_label = torch.ones(batch_size, 1, 47, 47).to(device)  # Adjusted for 800x800 output
        fake_label = torch.zeros(batch_size, 1, 47, 47).to(device)

        for i, disc in enumerate(self.discriminators):
            disc.zero_grad()
            if i == 0:
                real_output = disc(mobil_batch)
                fake_mobil = self.generators[0](ref_batch)
                fake_output = disc(fake_mobil.detach())
                d_loss_real = self.criterion(real_output, real_label)
                d_loss_fake = self.criterion(fake_output, fake_label)
                d_loss = (d_loss_real + d_loss_fake) * 0.5
            else:
                real_output = disc(ref_batch)
                fake_ref = self.generators[0](mobil_batch)
                fake_output = disc(fake_ref.detach())
                d_loss_real = self.criterion(real_output, real_label)
                d_loss_fake = self.criterion(fake_output, fake_label)
                d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            self.optimizers_d[i].step()

        for g in self.generators:
            g.zero_grad()
            fake_mobil = g(ref_batch)
            fake_ref = g(mobil_batch)
            d_mobil_output = self.discriminators[0](fake_mobil)
            d_ref_output = self.discriminators[1](fake_ref)
            g_loss_gan = self.criterion(d_mobil_output, real_label) + self.criterion(d_ref_output, real_label)
            cycle_loss = self.l1_loss(self.generators[1](fake_ref), mobil_batch) + self.l1_loss(self.generators[2](fake_mobil), ref_batch)
            identity_loss = self.l1_loss(g(mobil_batch), mobil_batch) + self.l1_loss(g(ref_batch), ref_batch)
            g_loss = g_loss_gan + 10.0 * cycle_loss + 5.0 * identity_loss
            g_loss.backward()
            self.optimizers_g[self.generators.index(g)].step()

        return g_loss.item(), d_loss.item()

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'generators_state_dict': [g.state_dict() for g in self.generators],
            'discriminators_state_dict': [d.state_dict() for d in self.discriminators],
            'optimizers_g_state_dict': [opt.state_dict() for opt in self.optimizers_g],
            'optimizers_d_state_dict': [opt.state_dict() for opt in self.optimizers_d]
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
        logger.info(f"Saved checkpoint at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        for i, g in enumerate(self.generators):
            g.load_state_dict(checkpoint['generators_state_dict'][i])
        for i, d in enumerate(self.discriminators):
            d.load_state_dict(checkpoint['discriminators_state_dict'][i])
        for i, opt in enumerate(self.optimizers_g):
            opt.load_state_dict(checkpoint['optimizers_g_state_dict'][i])
        for i, opt in enumerate(self.optimizers_d):
            opt.load_state_dict(checkpoint['optimizers_d_state_dict'][i])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def infer(self, mobil_image, output_dir="inferences"):
        os.makedirs(output_dir, exist_ok=True)
        self.generators[0].eval()
        with torch.no_grad():
            mobil_tensor = mobil_image.unsqueeze(0).to(device)
            enhanced_image = self.generators[0](mobil_tensor).cpu()
            enhanced_image = (enhanced_image + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            vutils.save_image(enhanced_image, os.path.join(output_dir, f'inference_{os.urandom(8).hex()}.png'))
            logger.info(f"Saved inference image to {output_dir}")

    def calculate_metrics(self, mobil_batch, ref_batch, enhanced_batch):
        mobil_np = mobil_batch.cpu().numpy().transpose((0, 2, 3, 1))
        ref_np = ref_batch.cpu().numpy().transpose((0, 2, 3, 1))
        enhanced_np = enhanced_batch.cpu().numpy().transpose((0, 2, 3, 1))
        
        batch_psnr = np.mean([psnr(ref_np[i], enhanced_np[i], data_range=1.0) for i in range(mobil_np.shape[0])])
        batch_ssim = np.mean([ssim(ref_np[i], enhanced_np[i], multichannel=True, data_range=1.0, win_size=3) for i in range(mobil_np.shape[0])])
        return batch_psnr, batch_ssim

    def train(self, epochs=50):
        transform = transforms.Compose([
            transforms.Resize((800, 800)),  # Updated to 800x800
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        datasets = {}
        for split in ["train", "val", "test"]:
            dataset = AstroDataset(self.mobil_dir, self.ref_dir, split, transform)
            if len(dataset) > 0:
                datasets[split] = dataset
            else:
                logger.warning(f"Skipping empty dataset for split {split}")

        if not datasets:
            logger.error("No valid datasets found, exiting training")
            return

        dataloaders = {split: DataLoader(datasets[split], batch_size=1, shuffle=True) for split in datasets}  # Reduced batch size to 1

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            for phase in dataloaders.keys():
                if phase == "train":
                    for g in self.generators:
                        g.train()
                    for d in self.discriminators:
                        d.train()
                else:
                    for g in self.generators:
                        g.eval()
                    for d in self.discriminators:
                        d.eval()

                running_g_loss = 0.0
                running_d_loss = 0.0
                for i, (mobil_batch, ref_batch) in enumerate(dataloaders[phase]):
                    mobil_batch, ref_batch = mobil_batch.to(device), ref_batch.to(device)
                    g_loss, d_loss = self.train_step(mobil_batch, ref_batch)
                    running_g_loss += g_loss
                    running_d_loss += d_loss

                    if phase == "train" and i % 10 == 0:
                        with torch.no_grad():
                            fake_mobil = self.generators[0](ref_batch)
                            psnr_val, ssim_val = self.calculate_metrics(mobil_batch, ref_batch, fake_mobil)
                            logger.info(f"Batch {i} - PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")

                epoch_g_loss = running_g_loss / len(dataloaders[phase])
                epoch_d_loss = running_d_loss / len(dataloaders[phase])
                logger.info(f"{phase} G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}")

                if phase == "train" and (epoch + 1) % 5 == 0:
                    self.save_checkpoint(epoch + 1)

            # Ensemble variance and inference on test set
            if "test" in dataloaders:
                with torch.no_grad():
                    test_mobil, _ = next(iter(dataloaders["test"]))
                    test_mobil = test_mobil.to(device)
                    preds = torch.stack([g(test_mobil) for g in self.generators])
                    variance = torch.var(preds, dim=0).mean().item()
                    logger.info(f"Ensemble Variance: {variance:.4f}")
                    self.infer(test_mobil[0])

# Main execution
if __name__ == "__main__":
    mobil_dir = r"D:/Shantanu/MBTR/sky-survey/MobilTelesco_Processed/crops"
    ref_dir = r"D:/Shantanu/MBTR/sky-survey/MobilTelesco_Processed/reference_images"
    gan = CMRA_GAN(mobil_dir, ref_dir)
    gan.train(epochs=50)