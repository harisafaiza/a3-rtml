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

sports.csv contains metadata for the dataset.
Images are stored in the /dataset/ directory.
🛠 Model Architecture

The model architecture utilizes Vision Transformer (ViT) for image classification, specifically the ViT-B/16 variant.

Base Model: ViT-B/16
Pretrained Weights: ViT_B_16_Weights.DEFAULT
Final Layer: Fully connected layer with 100 output classes
import torch
import torch.nn as nn
from torchvision.models import vit_b_16 as ViT, ViT_B_16_Weights

# Load the Vision Transformer model
model = ViT(weights=ViT_B_16_Weights.DEFAULT)
model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=100, bias=True))
🖼️ Data Augmentation & Preprocessing

To improve model generalization, the training pipeline includes several image augmentations:

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

Here are the key performance metrics of the trained model:

Metric	Value (%)
Training Accuracy	97.58%
Validation Accuracy	92.46%
Test Accuracy	98.61%
Loss	18.87
The model performed well in classifying various sports but had minor misclassifications in visually similar categories.

🚀 How to Run

Follow the steps below to train and evaluate the model:

1️⃣ Install Dependencies
First, install the required dependencies:

pip install torch torchvision pandas numpy matplotlib
2️⃣ Train the Model
To train the model, run:

python train.py
3️⃣ Evaluate the Model
After training, evaluate the model using:

python evaluate.py
🔗 Future Improvements

Use a larger dataset for better generalization.
Experiment with other models like Swin Transformer and ConvNeXt.
Fine-tune hyperparameters to further improve accuracy.
