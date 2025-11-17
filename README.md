# DC-UNet for Atheroma Segmentation

This project implements a DC-UNet (Dense Convolutional UNet) architecture for atheroma segmentation in medical images. The model utilizes a U-Net-based architecture with Dense Convolutional blocks to improve feature capture at different scales.

## 📋 About the Project

This project is a key component of **GigaSistêmica**, a collaborative initiative between GigaCandanga and the University of Brasília. GigaSistêmica aims to revolutionize diagnostic and predictive capabilities for systemic diseases through the integration of AI and medical imaging technologies.

This specific implementation focuses on **atheroma detection and segmentation**, providing an automated tool to assist medical professionals in identifying and analyzing atherosclerotic plaques in medical images.

## 🏗️ Architecture

The DC-UNet architecture is based on the U-Net framework with the following improvements:

- **Encoder**: DCBlock (Dense Convolutional) blocks with ResPath to preserve information
- **Decoder**: Upsampling with skip connections
- **Loss Function**: Supports IoU Loss and Focal Loss
- **Input**: Grayscale images (1 channel) or RGB (3 channels)
- **Output**: Binary segmentation mask

### Reference

This implementation is based on the DC-UNet architecture proposed in:

**DC-UNet: Rethinking the U-Net Architecture with Dual Channel Efficient CNN for Medical Images Segmentation**

*Ange Lou, Shuyue Guan, Murray Loew*

> Recently, deep learning has become much more popular in computer vision area. The Convolution Neural Network (CNN) has brought a breakthrough in images segmentation areas, especially, for medical images. In this regard, U-Net is the predominant approach to medical image segmentation task. The U-Net not only performs well in segmenting multimodal medical images generally, but also in some tough cases of them. However, we found that the classical U-Net architecture has limitation in several aspects. Therefore, we applied modifications: 1) designed efficient CNN architecture to replace encoder and decoder, 2) applied residual module to replace skip connection between encoder and decoder to improve based on the-state-of-the-art U-Net model. Following these modifications, we designed a novel architecture--DC-UNet, as a potential successor to the U-Net architecture. We created a new effective CNN architecture and build the DC-UNet based on this CNN. We have evaluated our model on three datasets with tough cases and have obtained a relative improvement in performance of 2.90%, 1.49% and 11.42% respectively compared with classical U-Net. In addition, we used the Tanimoto similarity to replace the Jaccard similarity for gray-to-gray image comparisons.

## 📁 Project Structure

```
gigasistemica-unet/
├── src/
│   ├── models/           # Model architectures
│   │   └── DC_UNet.py    # DC-UNet implementation
│   ├── data/             # Data loaders and datasets
│   │   ├── dataloader.py # Main data loaders
│   │   └── ateroma_dataloader.py  # Atheroma-specific data loader
│   ├── training/         # Training scripts
│   │   └── train.py      # Main training script
│   └── utils/            # Utilities
│       ├── loss.py       # Loss functions
│       ├── utils.py      # Helper functions
│       ├── validate.py   # Validation function
│       └── TTA.py        # Test Time Augmentation
├── config/               # Configuration files
│   └── config.py         # Centralized configuration file
├── scripts/              # Execution scripts
│   └── test.py           # Testing/validation script
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gigasistemica-unet
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

All configurations are centralized in the `config/config.py` file. You can:

1. **Edit the `config/config.py` file directly** to adjust paths and hyperparameters

2. **Use environment variables** (recommended for different environments):
```bash
export DATASET_DIR="/path/to/dataset"
export RUNS_DIR="/path/to/runs"
export DEVICE="cuda:0"
export BATCH_SIZE=4
export NUM_EPOCHS=300
export LEARNING_RATE=1e-4
```

### Expected Dataset Structure

The dataset should follow this structure:
```
dataset/
├── images/
│   ├── train/     # Training images (.jpg or .png)
│   └── val/       # Validation images (.jpg or .png)
└── masks/
    ├── train/     # Training masks (.png)
    └── val/       # Validation masks (.png or .tif)
```

## 📊 Usage

### Training

To train the model, run:

```bash
python src/training/train.py
```

Or with custom environment variables:
```bash
DATASET_DIR="/path/dataset" DEVICE="cuda:0" python src/training/train.py
```

The script will:
- Create a run directory with timestamp in `runs/`
- Save checkpoints periodically
- Log metrics to TensorBoard
- Run validation at each epoch

### Testing/Validation

To test a trained model:

```bash
python scripts/test.py --checkpoint /path/to/checkpoint.pth.tar --save_images
```

Available options:
- `--checkpoint`: Path to model checkpoint (required)
- `--val_img_dir`: Directory with validation images (optional, uses config)
- `--val_mask_dir`: Directory with validation masks (optional, uses config)
- `--device`: Execution device (`cuda` or `cpu`)
- `--apply_tta`: Apply Test Time Augmentation
- `--save_images`: Save result images
- `--output_dir`: Directory to save detailed metrics (CSV)

Complete example:
```bash
python scripts/test.py \
    --checkpoint runs/2024-11-17_15-30-00/checkpoint.pth.tar \
    --save_images \
    --apply_tta \
    --output_dir results/
```

## 🎯 Key Hyperparameters

- **TRAIN_SIZE**: Input image size (default: 512x512)
- **IN_CHANNELS**: Number of input channels (default: 1 for grayscale)
- **BATCH_SIZE**: Batch size (default: 4)
- **LEARNING_RATE**: Learning rate (default: 1e-4)
- **NUM_EPOCHS**: Number of epochs (default: 300)
- **LOSS_FUNCTION**: Loss function ('IoU' or 'Focal Loss')

## 📈 Metrics

The model calculates the following metrics:

- **Precision**: Segmentation precision
- **Recall**: Segmentation recall
- **F1 Score**: Harmonic mean of precision and recall
- **IoU (Intersection over Union)**: Overlap between prediction and ground truth
- **Dice Score**: Dice coefficient
- **AUC**: Area under the ROC curve

## 🔧 Features

### Test Time Augmentation (TTA)

The project supports TTA to improve prediction robustness. When enabled, the model makes predictions on multiple augmented versions of the image and combines the results.

### Data Augmentation

During training, the following transformations can be applied:
- Random rotation (up to 90 degrees)
- Horizontal and vertical flips
- Brightness and contrast adjustments
- Elastic transformations

## 📝 Notes

- The project was developed for atheroma segmentation but can be adapted for other semantic segmentation tasks
- Hardcoded paths have been removed and centralized in `config/config.py`
- Code has been organized into modules for easier maintenance and extension

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under [specify license].

## 👥 Authors

- **Matheus Virgílio Ferreira** - Initial development

## 🙏 Acknowledgments

- Based on the original U-Net architecture
- DC-UNet architecture from: Lou, A., Guan, S., & Loew, M. (2020). DC-UNet: Rethinking the U-Net Architecture with Dual Channel Efficient CNN for Medical Images Segmentation. 
- Uses components from PyTorch and open-source libraries
- Part of the GigaSistêmica initiative by GigaCandanga and University of Brasília
