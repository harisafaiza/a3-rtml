Sports Classification using Vision Transformers (ViT)

This project implements a deep learning model using Vision Transformers (ViT) for sports classification. The model is trained on an image dataset, leveraging PyTorch and torchvision for training and evaluation.

📂 Project Structure

RTML_A3/
│── dataset/
│   ├── sports.csv        # Sports dataset
│   ├── class_labels.csv  # Class mapping file (if applicable)
│── models/
│   ├── Ep.7.pth          # Trained model checkpoint
│── content/
│   ├── dataset.py        # Dataset loader and preprocessing
│── transformers.ipynb    # Main notebook for training & evaluation
│── train.py              # Training script
│── evaluate.py           # Model evaluation script
│── README.md             # Project documentation
📊 Dataset

The dataset consists of labeled sports images categorized into different sports.

sports.csv includes metadata for the dataset.
Images are stored in /dataset/ directory.
🛠 Model Architecture

Base Model: ViT-B/16 (Vision Transformer)
Pretrained Weights: ViT_B_16_Weights.DEFAULT
Final Layer: Fully connected layer with 100 output classes
import torch
import torch.nn as nn
from torchvision.models import vit_b_16 as ViT, ViT_B_16_Weights

# Load the Vision Transformer model
model = ViT(weights=ViT_B_16_Weights.DEFAULT)
model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=100, bias=True))
🖼️ Data Augmentation & Preprocessing

The training pipeline includes image augmentations for better generalization:

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.ToPILImage(mode='RGB'),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
📈 Results & Accuracy

Metric	Value (%)
Training Accuracy	97.58%
Validation Accuracy	92.46%
Test Accuracy	98.61%
Loss	18.87
The model performed well in classifying different sports but had minor misclassifications in visually similar categories.

🚀 How to Run

1️⃣ Install Dependencies
pip install torch torchvision pandas numpy matplotlib
2️⃣ Train the Model
python train.py
3️⃣ Evaluate the Model
python evaluate.py
🔗 Future Improvements

Use a larger dataset for better generalization
Experiment with Swin Transformer and ConvNeXt
Fine-tune hyperparameters to improve accuracy
