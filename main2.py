import os
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import torchvision
from torch import nn
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------------------
# Sentinel-2 Normalization
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
# Custom Dataset Class
# -------------------------------
class CustomSEN12MSDataset(Dataset):
    def __init__(self, images_dir, labels_dir, patch_size=256):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.patch_size = patch_size
        self.samples = []

        for root, _, files in os.walk(images_dir):
            for filename in files:
                if filename.endswith(".tif") and "_s2_" in filename:
                    image_path = os.path.join(root, filename)

                    # Trocar "_s2_" por "_lc_" para localizar o label correto
                    label_filename = filename.replace("_s2_", "_lc_")

                    # Pega o nome da subpasta onde está o s2
                    s2_folder = os.path.basename(root)
                    lc_folder = s2_folder.replace("s2_", "lc_")

                    label_path = os.path.join(labels_dir, lc_folder, label_filename)

                    if os.path.exists(label_path):
                        self.samples.append((image_path, label_path))
                    else:
                        print(f"Aviso: Label não encontrado para {image_path} -> {label_path}")

        print(f"Total de amostras válidas encontradas: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Read Sentinel-2 image
        with rasterio.open(img_path) as src:
            img = src.read()  # Shape: (13, H, W)
            img = torch.from_numpy(img).float() / 10000.0  # Scale to 0-1 typical for Sentinel-2

        # Read label
        with rasterio.open(label_path) as src:
            lbl = src.read(1)  # Shape: (H, W), single channel
            lbl = torch.from_numpy(lbl).long()

        # Optional: center crop / random crop to patch_size
        h, w = lbl.shape
        if h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            img = img[:, top:top+self.patch_size, left:left+self.patch_size]
            lbl = lbl[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Skip images that are too small
            raise ValueError(f"Image {img_path} too small for patch size {self.patch_size}")

        return img, lbl

# -------------------------------
# Lightning Module (unchanged)
# -------------------------------
def get_deeplab(num_classes):
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(13, old_conv.out_channels, old_conv.kernel_size,
                         old_conv.stride, old_conv.padding, bias=old_conv.bias is not None)

    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:] = avg_weight.repeat(1, 10, 1, 1)

    model.backbone.conv1 = new_conv
    model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
    return model

class SegmentationModule(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = get_deeplab(num_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=17)

    def forward(self, x):
        return self.model(x)['out']

    def training_step(self, batch, batch_idx):
        img, lbl = batch
        img = (img - S2_MEAN.to(img.device)) / S2_STD.to(img.device)
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
    preds = torch.argmax(F.softmax(logits, dim=0), dim=0).cpu().numpy()
    rgb = image[:3].cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title('Input RGB')

    plt.subplot(1, 2, 2)
    plt.imshow(preds, cmap='tab20')
    plt.title('Prediction')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # -------------------------------
    # DataLoader Setup
    # -------------------------------
    root_path = "./train"
    train_dataset = CustomSEN12MSDataset(
        images_dir=os.path.join(root_path, "data"),
        labels_dir=os.path.join(root_path, "labels")
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    # Optionally create a validation dataset too
    val_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)

    # -------------------------------
    # Training
    # -------------------------------
    model = SegmentationModule(num_classes=18)
    trainer = pl.Trainer(accelerator="cpu", max_epochs=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # -------------------------------
    # Visualize sample
    # -------------------------------
    sample_img, sample_lbl = next(iter(train_loader))
    sample_img = sample_img[0]
    sample_lbl = sample_lbl[0]

    with torch.no_grad():
        input_img = (sample_img.unsqueeze(0).cuda() - S2_MEAN.cuda()) / S2_STD.cuda()
        logits = model(input_img).squeeze()

    visualize_prediction(sample_img, logits)
