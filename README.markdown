# StrCGAN
![Inferences](https://github.com/Shantanu-Parmar/StrCGAN/blob/ebd391fc39b399cfa1dcffc317f52487a971e5f5/Aldebaran_inf.gif)
## Overview
StrCGAN (Stellar Cyclic Generative Adversarial Network) is an advanced deep learning framework designed for the restoration and enhancement of low-resolution astrophotography images, with a focus on reconstructing high-fidelity representations of celestial objects. Tailored for datasets like MobilTelesco—a smartphone-based astrophotography collection—StrCGAN addresses the challenges of limited resolution, atmospheric distortion, and feature sparsity, where critical stellar morphologies (e.g., stars, galaxies, nebulae) are obscured by background noise. Building on the foundation of CycleGAN, StrCGAN extends traditional 2D image-to-image translation with innovative features: 3D convolutional layers to capture volumetric spatial correlations, multi-spectral fusion to align optical and near-infrared (NIR) domains, and astrophysical regularization modules to preserve stellar morphology. Guided by ground-truth references from multi-mission all-sky surveys (spanning optical to NIR), StrCGAN produces visually sharper and physically consistent reconstructions, outperforming standard GAN models in astrophysical image enhancement.
The framework leverages a multi-resolution attention mechanism to focus on sparse, low-contrast features, enhancing reconstruction quality efficiently. This makes StrCGAN particularly suited for astronomical analysis, where high-fidelity imaging is crucial for tracking cosmic evolution and identifying stellar details.

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
