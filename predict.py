import csv
import os
import shutil

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnext101_32x8d


# ==========================================
# 1. Configuration and Paths
# ==========================================
save_dir = '/content/drive/MyDrive/Colab_Files/CV_HW1'
weights_path = os.path.join(save_dir, 'se_resnext101_advanced_weights.pth')

num_classes = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. Define Squeeze-and-Excitation (SE) Block
# ==========================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Must perfectly match the architecture used during training.
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
# 3. Initialize Model and Load Weights
# ==========================================
model = resnext101_32x8d(weights=None)

# Insert SE-Blocks (matching training architecture)
model.layer1 = nn.Sequential(model.layer1, SEBlock(256))
model.layer2 = nn.Sequential(model.layer2, SEBlock(512))
model.layer3 = nn.Sequential(model.layer3, SEBlock(1024))
model.layer4 = nn.Sequential(model.layer4, SEBlock(2048))

# Modify classification head
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load weights
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
else:
    raise FileNotFoundError(f"Weight file not found: {weights_path}")

model = model.to(device)
model.eval()


# ==========================================
# 4. Data Preprocessing and Dataset
# ==========================================
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        valid_extensions = ('.png', '.jpg', '.jpeg')
        self.image_files = [
            f for f in os.listdir(root_dir) if f.lower().endswith(valid_extensions)
        ]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        file_name_without_ext = os.path.splitext(img_name)[0]
        return image, file_name_without_ext


# ==========================================
# 5. Inference with Test-Time Augmentation (TTA)
# ==========================================
test_dir = './dataset_folder/data/test'

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")

test_dataset = TestDataset(root_dir=test_dir, transform=val_transform)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2
)

class_names = sorted([str(i) for i in range(num_classes)])
predictions = []

with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)
        
        # TTA: Original image prediction
        outputs_orig = model(images)
        
        # TTA: Flipped image prediction
        images_flipped = torch.flip(images, dims=[3])
        outputs_flipped = model(images_flipped)
        
        # TTA: Average predictions
        outputs_final = (outputs_orig + outputs_flipped) / 2.0
        _, preds = torch.max(outputs_final, 1)

        for i in range(len(img_names)):
            pred_idx = preds[i].item()
            pred_label = class_names[pred_idx]
            predictions.append([img_names[i], pred_label])

# ==========================================
# 6. Export Predictions
# ==========================================
submission_path = 'prediction.csv'
with open(submission_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'pred_label'])
    writer.writerows(predictions)

try:
    os.makedirs(save_dir, exist_ok=True)
    drive_submission_path = os.path.join(save_dir, 'prediction_tta.csv')
    shutil.copy(submission_path, drive_submission_path)
    print(f"Predictions successfully saved and backed up to: {drive_submission_path}")
except Exception as e:
    print(f"Error copying submission to Google Drive: {e}")
