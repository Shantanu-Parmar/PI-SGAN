# StrCGAN

## Overview
StrCGAN (Structural Conditional Generative Adversarial Network) is a sophisticated deep learning framework designed for image-to-image translation and enhancement, with a particular focus on astronomical and medical imaging tasks. This framework leverages the power of Generative Adversarial Networks (GANs) to transform low-resolution or noisy images—such as those captured by the Mobil telescope—into high-resolution or enhanced versions by incorporating reference data. The architecture emphasizes structural consistency and employs attention mechanisms to prioritize critical image features, making it highly effective for applications requiring precise detail preservation and quality improvement.

GANs operate on a two-player game principle involving a generator and a discriminator. The generator synthesizes images from random noise or input data, while the discriminator evaluates their authenticity against real images. In StrCGAN, this adversarial process is conditioned on structural and reference information, enabling the model to learn complex mappings while maintaining fidelity to the input’s inherent structure. This approach is particularly valuable in scenarios where traditional enhancement techniques fail to capture fine details or contextual relationships, such as in celestial object imaging or medical diagnostics.

## Features
- **Conditional GAN Architecture:** Utilizes multiple generators and discriminators to enforce cycle consistency and identity preservation.
- **Attention Mechanism:** Incorporates attention layers to focus on relevant structural details during image generation.
- **Metric Evaluation:** Computes PSNR, SSIM, FID, Inception Score, LPIPS, and confidence metrics for quantitative assessment.
- **Checkpoint Support:** Allows saving and resuming training from checkpoints for seamless experimentation.
- **Inference Capability:** Generates enhanced images and saves detailed metrics for analysis.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StrCGAN.git
   cd StrCGAN
   ```
2. Install required dependencies:
   ```bash
   pip install torch torchvision lpips scikit-image scipy pandas
   ```
3. Ensure you have a compatible NVIDIA GPU with CUDA installed for optimal performance (optional but recommended).

## Usage
### Dataset Preparation
- Place your input images (e.g., Mobil telescope crops) in `MobilTelesco_Processed/crops/train`, `val`, and `test` subdirectories.
- Place corresponding reference images in `MobilTelesco_Processed/augmented_ground_truths` with matching object names.
- Ensure images are in `.jpg` (input) and `.png` (reference) formats.

### Training
To train the model from scratch or resume from a checkpoint:
```bash
python StrCGAN.py
```
- Checkpoints are saved every 5 epochs in the `checkpoints` directory.
- To resume from a specific checkpoint (e.g., `checkpoints/best_model.pth`), edit `train_cmra_gan.py` to include `gan.load_checkpoint("checkpoints/best_model.pth")` before `gan.train(epochs=...)`.

### Inference
To run inference with a trained model:
- Modify `StrCGAN.py` to call `gan.infer(dataloader)` with the test dataset.
- Run the script:
  ```bash
  python StrCGAN.py
  ```
- Results are saved in the `inferences` directory, with metrics in `test_metrics.csv`.

## Directory Structure
```
StrCGAN/
├── MobilTelesco_Processed/
│   ├── crops/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── augmented_ground_truths/
├── checkpoints/         # Saved model checkpoints
├── inferences/          # Inference output images
├── train_visuals/       # Training visualization images
├── val_visuals/         # Validation visualization images
├── StrCGAN.py    # Main script
└── README.md            # This file
```

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes. Open issues for discussions or feature requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built using PyTorch and torchvision libraries.
- Inspired by advancements in GAN research for image enhancement and translation tasks.