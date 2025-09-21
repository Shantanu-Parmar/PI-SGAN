import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils
from PIL import Image
import numpy as np
import logging
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.utils as vutils
from scipy.linalg import sqrtm
from scipy.stats import entropy
import pandas as pd
from torchvision.models import inception_v3
from lpips import LPIPS
from torchvision.utils import save_image

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
        self.object_names = []

        if os.path.exists(self.mobil_dir):
            for obj in os.listdir(self.mobil_dir):
                mobil_obj_dir = os.path.join(self.mobil_dir, obj)
                if os.path.isdir(mobil_obj_dir):
                    self.mobil_images.extend([
                        os.path.join(mobil_obj_dir, x)
                        for x in os.listdir(mobil_obj_dir)
                        if x.lower().endswith(".jpg")
                    ])
                    self.object_names.extend([obj] * len(os.listdir(mobil_obj_dir)))
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
        self.object_names = self.object_names[:len(self.mobil_images)]  # Align with image count

    def __len__(self):
        return min(len(self.mobil_images), len(self.ref_images))

    def __getitem__(self, idx):
        mobil_img = Image.open(self.mobil_images[idx % len(self.mobil_images)]).convert("RGB")
        ref_img = Image.open(self.ref_images[idx % len(self.ref_images)]).convert("RGB")
        obj_name = self.object_names[idx % len(self.object_names)]

        if self.transform:
            mobil_img = self.transform(mobil_img)
            ref_img = self.transform(ref_img)

        return mobil_img, ref_img, obj_name
    
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mobil_dir = mobil_dir
        self.ref_dir = ref_dir
        self.generators = [Generator().to(self.device) for _ in range(3)]
        self.discriminators = [Discriminator().to(self.device) for _ in range(2)]
        self.optimizers_g = [optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999)) for g in self.generators]
        self.optimizers_d = [optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999)) for d in self.discriminators]
        self.criterion = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device).eval()
        self.lpips_model = LPIPS(net='alex').to(self.device)
        self.best_val_loss = float('inf')

    def train_step(self, mobil_batch, ref_batch):
        batch_size = mobil_batch.size(0)
        real_label = torch.ones(batch_size, 1, 24, 24).to(self.device)
        fake_label = torch.zeros(batch_size, 1, 24, 24).to(self.device)

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

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'generators_state_dict': [g.state_dict() for g in self.generators],
            'discriminators_state_dict': [d.state_dict() for d in self.discriminators],
            'optimizers_g_state_dict': [opt.state_dict() for opt in self.optimizers_g],
            'optimizers_d_state_dict': [opt.state_dict() for opt in self.optimizers_d],
            'best_val_loss': self.best_val_loss
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
        logger.info(f"Saved checkpoint at epoch {epoch}{' (best model)' if is_best else ''}")

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
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def infer(self, dataloader):
        self.generators[0].eval()
        metrics_df = pd.DataFrame(columns=['Object', 'PSNR', 'SSIM', 'FID', 'Inception_Score', 'LPIPS', 'Confidence'])
        output_dir = "inferences"
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for i, (mobil_batch, ref_batch, obj_name) in enumerate(dataloader):
                mobil_batch, ref_batch = mobil_batch.to(self.device), ref_batch.to(self.device)
                enhanced_batch = self.generators[0](mobil_batch)
                ref_batch = (ref_batch.cpu() + 1) / 2
                mobil_batch = (mobil_batch.cpu() + 1) / 2
                enhanced_batch = (enhanced_batch.cpu() + 1) / 2

                psnr_val, ssim_val = self.calculate_metrics(mobil_batch, ref_batch, enhanced_batch)
                fid_val = self.calculate_fid(ref_batch, enhanced_batch)
                is_val = self.calculate_inception_score(enhanced_batch)
                lpips_val = self.calculate_lpips(ref_batch.to(self.device), enhanced_batch.to(self.device))
                confidence = self.discriminators[0](enhanced_batch.to(self.device)).mean().item()

                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Object': [obj_name[0]], 'PSNR': [psnr_val], 'SSIM': [ssim_val],
                    'FID': [fid_val], 'Inception_Score': [is_val], 'LPIPS': [lpips_val], 'Confidence': [confidence]
                })], ignore_index=True)

                grid = utils.make_grid([ref_batch[0], mobil_batch[0], enhanced_batch[0]], nrow=3, normalize=True)
                utils.save_image(grid, os.path.join(output_dir, f'{obj_name[0]}_inference_{i}_{os.urandom(8).hex()}.png'))
                logger.info(f"Saved inference image for {obj_name[0]}")

        metrics_df.to_csv("test_metrics.csv", index=False)
        logger.info(f"Inference metrics saved to test_metrics.csv")

    def calculate_metrics(self, mobil_batch, ref_batch, enhanced_batch):
        mobil_np = mobil_batch.cpu().numpy().transpose((0, 2, 3, 1))
        ref_np = ref_batch.cpu().numpy().transpose((0, 2, 3, 1))
        enhanced_np = enhanced_batch.cpu().numpy().transpose((0, 2, 3, 1))
        
        batch_psnr = np.mean([psnr(ref_np[i], enhanced_np[i], data_range=1.0) for i in range(mobil_np.shape[0])])
        batch_ssim = np.mean([ssim(ref_np[i], enhanced_np[i], multichannel=True, data_range=1.0, win_size=3) for i in range(mobil_np.shape[0])])
        return batch_psnr, batch_ssim

    def calculate_fid(self, real_images, generated_images):
        real_images = (real_images + 1) / 2
        generated_images = (generated_images + 1) / 2

        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def tensor_to_pil_batch(tensors):
            pil_images = []
            for tensor in tensors:
                tensor = tensor.permute(1, 2, 0).cpu().numpy()
                pil_img = Image.fromarray((tensor * 255).astype(np.uint8))
                pil_images.append(transform(pil_img))
            return torch.stack(pil_images).to(self.device)

        real_images_processed = tensor_to_pil_batch(real_images)
        generated_images_processed = tensor_to_pil_batch(generated_images)

        def get_features(images):
            with torch.no_grad():
                features = self.inception_model(images).view(images.size(0), -1)
            return features

        real_features = get_features(real_images_processed)
        generated_features = get_features(generated_images_processed)

        mu1, sigma1 = real_features.mean(0), np.cov(real_features.cpu().numpy().T)
        mu2, sigma2 = generated_features.mean(0), np.cov(generated_features.cpu().numpy().T)

        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid.item()

    def calculate_inception_score(self, generated_images):
        generated_images = (generated_images + 1) / 2

        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def tensor_to_pil_batch(tensors):
            pil_images = []
            for tensor in tensors:
                tensor = tensor.permute(1, 2, 0).cpu().numpy()
                pil_img = Image.fromarray((tensor * 255).astype(np.uint8))
                pil_images.append(transform(pil_img))
            return torch.stack(pil_images).to(self.device)

        generated_images_processed = tensor_to_pil_batch(generated_images)

        with torch.no_grad():
            preds = self.inception_model(generated_images_processed)
            preds = nn.functional.softmax(preds, dim=1)

        kl_divs = [entropy(pred.cpu().numpy(), preds.mean(0).cpu().numpy()) for pred in preds]
        is_score = np.exp(np.mean(kl_divs))
        return is_score

    def calculate_lpips(self, ref_batch, enhanced_batch):
        ref_images = ((ref_batch + 1) / 2).to(self.device)
        enhanced_images = ((enhanced_batch + 1) / 2).to(self.device)
        return self.lpips_model(ref_images, enhanced_images).mean().item()

    def validate(self, dataloader, epoch):
        self.generators[0].eval()
        self.discriminators[0].eval()
        metrics_df = pd.DataFrame(columns=['Object', 'Epoch', 'PSNR', 'SSIM', 'FID', 'Inception_Score', 'LPIPS', 'Confidence'])
        output_dir = "val_visuals"
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for i, (mobil_batch, ref_batch, obj_name) in enumerate(dataloader):
                mobil_batch, ref_batch = mobil_batch.to(self.device), ref_batch.to(self.device)
                enhanced_batch = self.generators[0](mobil_batch)
                ref_batch = (ref_batch.cpu() + 1) / 2
                mobil_batch = (mobil_batch.cpu() + 1) / 2
                enhanced_batch = (enhanced_batch.cpu() + 1) / 2

                psnr_val, ssim_val = self.calculate_metrics(mobil_batch, ref_batch, enhanced_batch)
                fid_val = self.calculate_fid(ref_batch, enhanced_batch)
                is_val = self.calculate_inception_score(enhanced_batch)
                lpips_val = self.calculate_lpips(ref_batch.to(self.device), enhanced_batch.to(self.device))
                confidence = self.discriminators[0](enhanced_batch.to(self.device)).mean().item()

                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Object': [obj_name[0]], 'Epoch': [epoch], 'PSNR': [psnr_val], 'SSIM': [ssim_val],
                    'FID': [fid_val], 'Inception_Score': [is_val], 'LPIPS': [lpips_val], 'Confidence': [confidence]
                })], ignore_index=True)

                grid = utils.make_grid([ref_batch[0], mobil_batch[0], enhanced_batch[0]], nrow=3, normalize=True)
                utils.save_image(grid, os.path.join(output_dir, f'{obj_name[0]}_val_epoch_{epoch}_{i}_{os.urandom(8).hex()}.png'))
                logger.info(f"Saved validation image for {obj_name[0]} at epoch {epoch}")

        metrics_df.to_csv("val_metrics.csv", mode='a', index=False)
        logger.info(f"Validation metrics saved to val_metrics.csv for epoch {epoch}")

    def train(self, epochs=100):
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
                    metrics_df = pd.DataFrame(columns=['Object', 'Epoch', 'PSNR', 'SSIM', 'FID', 'Inception_Score', 'LPIPS', 'Confidence'])
                    output_dir = "train_visuals"
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    for g in self.generators:
                        g.eval()
                    for d in self.discriminators:
                        d.eval()

                running_g_loss = 0.0
                running_d_loss = 0.0
                for i, (mobil_batch, ref_batch, obj_name) in enumerate(dataloaders[phase]):
                    mobil_batch, ref_batch = mobil_batch.to(self.device), ref_batch.to(self.device)
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

                if phase == "train":
                    with torch.no_grad():
                        for i, (mobil_batch, ref_batch, obj_name) in enumerate(dataloaders["train"]):
                            mobil_batch, ref_batch = mobil_batch.to(self.device), ref_batch.to(self.device)
                            enhanced_batch = self.generators[0](mobil_batch)
                            ref_batch = (ref_batch.cpu() + 1) / 2
                            mobil_batch = (mobil_batch.cpu() + 1) / 2
                            enhanced_batch = (enhanced_batch.cpu() + 1) / 2

                            psnr_val, ssim_val = self.calculate_metrics(mobil_batch, ref_batch, enhanced_batch)
                            fid_val = self.calculate_fid(ref_batch, enhanced_batch)
                            is_val = self.calculate_inception_score(enhanced_batch)
                            lpips_val = self.calculate_lpips(ref_batch.to(self.device), enhanced_batch.to(self.device))
                            confidence = self.discriminators[0](enhanced_batch.to(self.device)).mean().item()

                            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                                'Object': [obj_name[0]], 'Epoch': [epoch], 'PSNR': [psnr_val], 'SSIM': [ssim_val],
                                'FID': [fid_val], 'Inception_Score': [is_val], 'LPIPS': [lpips_val], 'Confidence': [confidence]
                            })], ignore_index=True)

                            grid = utils.make_grid([ref_batch[0], mobil_batch[0], enhanced_batch[0]], nrow=3, normalize=True)
                            utils.save_image(grid, os.path.join(output_dir, f'{obj_name[0]}_train_epoch_{epoch}_{i}_{os.urandom(8).hex()}.png'))
                            logger.info(f"Saved training image for {obj_name[0]} at epoch {epoch}")

                    metrics_df.to_csv("train_metrics.csv", mode='a', index=False)
                    logger.info(f"Training metrics saved to train_metrics.csv for epoch {epoch}")

            if "val" in dataloaders:
                self.validate(dataloaders["val"], epoch)

        # Inference on test after training with best model
        if "test" in dataloaders:
            best_checkpoint = os.path.join(self.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_checkpoint):
                self.load_checkpoint(best_checkpoint)
                self.infer(dataloaders["test"])
            else:
                logger.warning("No best model found, using last checkpoint")
                self.infer(dataloaders["test"])

# Main execution
if __name__ == "__main__":
    mobil_dir = "MobilTelesco_Processed/crops"
    ref_dir = "MobilTelesco_Processed/reference_images"
    gan = CMRA_GAN(mobil_dir, ref_dir)
    gan.train(epochs=100)