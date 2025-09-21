import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import logging
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.utils as vutils
from scipy.linalg import sqrtm

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
                ref_obj_dir = os.path.join(ref_dir, obj)
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

        # Initialize Inception model for FID and IP
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()

    def train_step(self, mobil_batch, ref_batch):
        batch_size = mobil_batch.size(0)
        real_label = torch.ones(batch_size, 1, 24, 24).to(device)
        fake_label = torch.zeros(batch_size, 1, 24, 24).to(device)

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

    def calculate_fid(self, real_images, generated_images):
        # Denormalize tensors from [-1, 1] to [0, 1]
        real_images = (real_images + 1) / 2
        generated_images = (generated_images + 1) / 2

        # Resize to 299x299 for Inception v3
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Convert tensors to PIL images and apply transform
        def tensor_to_pil_batch(tensors):
            pil_images = []
            for tensor in tensors:
                # Remove batch dimension and convert to PIL
                tensor = tensor.permute(1, 2, 0).cpu().numpy()  # Shape: H x W x C
                pil_img = Image.fromarray((tensor * 255).astype(np.uint8))
                pil_images.append(transform(pil_img))
            return torch.stack(pil_images).to(device)

        real_images_processed = tensor_to_pil_batch(real_images)
        generated_images_processed = tensor_to_pil_batch(generated_images)

        # Extract features using Inception v3
        def get_features(images):
            with torch.no_grad():
                features = self.inception_model(images).view(images.size(0), -1)
            return features

        real_features = get_features(real_images_processed)
        generated_features = get_features(generated_images_processed)

        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(0), np.cov(real_features.cpu().numpy().T)
        mu2, sigma2 = generated_features.mean(0), np.cov(generated_features.cpu().numpy().T)

        # FID calculation
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid.item()

    def calculate_ip(self, real_images, generated_images):
        # Denormalize tensors from [-1, 1] to [0, 1]
        real_images = (real_images + 1) / 2
        generated_images = (generated_images + 1) / 2

        # Resize to 299x299 for Inception v3
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Convert tensors to PIL images and apply transform
        def tensor_to_pil_batch(tensors):
            pil_images = []
            for tensor in tensors:
                tensor = tensor.permute(1, 2, 0).cpu().numpy()  # Shape: H x W x C
                pil_img = Image.fromarray((tensor * 255).astype(np.uint8))
                pil_images.append(transform(pil_img))
            return torch.stack(pil_images).to(device)

        real_images_processed = tensor_to_pil_batch(real_images)
        generated_images_processed = tensor_to_pil_batch(generated_images)

        # Extract features and use a simple classifier (approximation)
        features_real = self.inception_model(real_images_processed).view(real_images_processed.size(0), -1)
        features_generated = self.inception_model(generated_images_processed).view(generated_images_processed.size(0), -1)
        
        # Simple precision: percentage of generated images closer to real distribution (approximation)
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(features_generated.cpu().numpy(), features_real.cpu().numpy())
        min_distances = np.min(distances, axis=1)
        threshold = np.mean(min_distances)  # Threshold based on average distance
        ip_score = np.mean(min_distances < threshold) * 100  # Percentage of generated within threshold
        return ip_score

    def train(self, epochs=50):
        transform = transforms.Compose([
            transforms.Resize((200, 200)),
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

        dataloaders = {split: DataLoader(datasets[split], batch_size=4, shuffle=True) for split in datasets}

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
                            fid_val = self.calculate_fid(ref_batch, fake_mobil)
                            ip_val = self.calculate_ip(ref_batch, fake_mobil)
                            logger.info(f"Batch {i} - PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, FID: {fid_val:.4f}, IP: {ip_val:.2f}%")

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
    mobil_dir = r"D:\Shantanu\MBTR\sky-survey\MobilTelesco_Processed\crops"
    ref_dir = r"D:\Shantanu\MBTR\sky-survey\MobilTelesco_Processed\reference_images"
    gan = CMRA_GAN(mobil_dir, ref_dir)
    gan.train(epochs=50)