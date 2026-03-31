import os
import tarfile

import gdown
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights


# ==========================================
# 1. Download and extract dataset
# ==========================================
file_id = '1vxiXJHUo6ZPGxBGXwrsSutOpqfJ6HN9D'
tar_path = 'dataset.tar'
extract_path = './dataset_folder'

if not os.path.exists(tar_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, tar_path, quiet=False)

if not os.path.exists(extract_path) or not os.listdir(extract_path):
    os.makedirs(extract_path, exist_ok=True)
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)

# ==========================================
# 2. Image preprocessing and data loading
# ==========================================
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

train_dir = os.path.join(extract_path, 'data/train')
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2
)

num_classes = len(train_dataset.classes)


# ==========================================
# 3. Define Squeeze-and-Excitation (SE) Block
# ==========================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Squeeze: Global Average Pooling.
    Excitation: Two fully connected layers to learn channel-wise weights.
    """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==========================================
# 4. Initialize and modify ResNeXt-101 backbone
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained weights
model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

# Insert SE-Blocks after each main layer
model.layer1 = nn.Sequential(model.layer1, SEBlock(256))
model.layer2 = nn.Sequential(model.layer2, SEBlock(512))
model.layer3 = nn.Sequential(model.layer3, SEBlock(1024))
model.layer4 = nn.Sequential(model.layer4, SEBlock(2048))

# Modify classification head
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ==========================================
# 5. Define loss function, optimizer, and scheduler
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 20
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ==========================================
# 6. Training loop with Automatic Mixed Precision (AMP)
# ==========================================
scaler = torch.amp.GradScaler('cuda')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    current_lr = optimizer.param_groups[0]['lr']

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

# ==========================================
# 7. Save model weights to Google Drive
# ==========================================
try:
    from google.colab import drive
    drive.mount('/content/drive')

    save_dir = '/content/drive/MyDrive/Colab_Files/CV_HW1'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'se_resnext101_advanced_weights.pth')
    torch.save(model.state_dict(), save_path)

    print(f"Model successfully saved to: {save_path}")

except Exception as e:
    print(f"Error occurred while saving the model: {e}")
