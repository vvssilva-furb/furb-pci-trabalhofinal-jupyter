# TODO: 
# add SAR inputs
# apply cloud masking using Sentinel-2 SCL or external cloud masks
# properly convert s2 to rgb

# Install dependencies (if not done already):
# pip install torchgeo rasterio torchvision torch pytorch-lightning matplotlib

import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import SEN12MS
from torchgeo.samplers import RandomGeoSampler
import torchvision
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# -------------------------------
# Sentinel-2 Normalization values (mean and std for each band)
# These values are approximate (commonly used for Sentinel-2 preprocessing)
# Source: BigEarthNet/SEN12MS community
# -------------------------------
S2_MEAN = torch.tensor([
    0.085, 0.09, 0.095, 0.1, 0.12, 0.15, 0.19,
    0.22, 0.25, 0.28, 0.3, 0.32, 0.35
]).view(1, 13, 1, 1)

S2_STD = torch.tensor([
    0.02, 0.02, 0.02, 0.02, 0.025, 0.03, 0.035,
    0.04, 0.045, 0.05, 0.055, 0.06, 0.065
]).view(1, 13, 1, 1)

# -------------------------------
# DataModule for SEN12MS
# -------------------------------
class SEN12MSSegDataModule(torch.utils.data.Dataset):
    def __init__(self, root, split, patch_size=256, batch_size=8):
        self.dataset = SEN12MS(root=root, split=split, bands=SEN12MS.BAND_SETS["s2-all"])
        self.sampler = RandomGeoSampler(self.dataset, size=patch_size, length=1000)
        self.batch_size = batch_size

    def dataloader(self):
        def collate_fn(batch):
            imgs, sars, labels = zip(*batch)
            imgs = torch.stack(imgs)
            labels = torch.stack(labels)
            return imgs, labels
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=4, collate_fn=collate_fn)

# -------------------------------
# Modified DeepLabV3 for 13 bands
# -------------------------------
def get_deeplab(num_classes):
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    
    # Modify first conv layer
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(
        in_channels=13,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    
    with torch.no_grad():
        # Copy weights for RGB channels
        new_conv.weight[:, :3] = old_conv.weight
        # Initialize extra channels by average of RGB weights
        avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:] = avg_weight.repeat(1, 10, 1, 1)
    
    model.backbone.conv1 = new_conv
    model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
    return model

# -------------------------------
# Lightning Module
# -------------------------------
class SegmentationModule(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = get_deeplab(num_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming class 0 is background

    def forward(self, x):
        return self.model(x)['out']

    def training_step(self, batch, batch_idx):
        img, lbl = batch
        img = (img - S2_MEAN.to(img.device)) / S2_STD.to(img.device)  # Normalize all 13 bands
        logits = self(img)
        loss = self.criterion(logits, lbl)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, lbl = batch
        img = (img - S2_MEAN.to(img.device)) / S2_STD.to(img.device)
        logits = self(img)
        loss = self.criterion(logits, lbl)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

# -------------------------------
# Visualization
# -------------------------------
def visualize_prediction(image, logits):
    # image: [13, H, W]
    # logits: [num_classes, H, W]

    # Get predicted class for each pixel
    preds = torch.argmax(F.softmax(logits, dim=0), dim=0).cpu().numpy()

    # Convert image tensor to numpy (show only RGB for visualization)
    rgb = image[:3].cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalize for display

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title('Input RGB (from 13-band image)')
    
    plt.subplot(1, 2, 2)
    plt.imshow(preds, cmap='tab20')  # Visualize predicted classes
    plt.title('Predicted Land Cover')
    
    plt.tight_layout()
    plt.show()

# -------------------------------
# Training Setup
# -------------------------------
root_path = "./train/ROIs1158_spring"  # <- Change this to your actual dataset path

train_module = SEN12MSSegDataModule(root=root_path, split="train")
val_module   = SEN12MSSegDataModule(root=root_path, split="test")

model = SegmentationModule(num_classes=6)

trainer = pl.Trainer(gpus=1, max_epochs=20)
trainer.fit(model,
            train_dataloaders=train_module.dataloader(),
            val_dataloaders=val_module.dataloader())

# -------------------------------
# Visualize a prediction from training data
# -------------------------------
sample_img, sample_lbl = next(iter(train_module.dataloader()))
sample_img = sample_img[0]  # Take first sample
sample_lbl = sample_lbl[0]

with torch.no_grad():
    input_img = (sample_img.unsqueeze(0) - S2_MEAN) / S2_STD
    logits = model(input_img.cuda()).squeeze()

visualize_prediction(sample_img, logits)
