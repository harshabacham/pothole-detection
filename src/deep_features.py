"""
src/deep_features.py
====================
Deep Learning feature extraction and Convolutional Autoencoder (CAE) utilities
using PyTorch. Provides pretrained ResNet embeddings and unsupervised surface
reconstruction anomaly scoring.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class DeepFeatureExtractor:
    """
    Extracts high-level visual feature embeddings from road patches using a
    pretrained, frozen ResNet-18 backbone.
    """

    def __init__(self, output_dim: int = 128, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load pretrained ResNet-18
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Truncate classification head, retain avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Optional linear projection to reduce embedding size
        self.projection = nn.Linear(512, output_dim)

        self.features.to(self.device)
        self.projection.to(self.device)
        self.features.eval()
        self.projection.eval()

        # Standard ImageNet normalization transform
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract_patch(self, patch_bgr: np.ndarray) -> np.ndarray:
        """
        Extract a 1D deep feature embedding vector for a single BGR patch.
        """
        if patch_bgr is None or patch_bgr.size == 0:
            return np.zeros(128, dtype=np.float32)

        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        feat = self.features(tensor)
        feat = torch.flatten(feat, 1)
        emb = self.projection(feat)

        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_batch(self, patches_bgr: list, batch_size: int = 32) -> np.ndarray:
        """
        Extract feature embeddings for a list of BGR patch images.
        """
        if not patches_bgr:
            return np.empty((0, 128), dtype=np.float32)

        embeddings = []
        for i in range(0, len(patches_bgr), batch_size):
            batch_slice = patches_bgr[i : i + batch_size]
            tensors = []
            for p in batch_slice:
                rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensors.append(self.transform(pil_img))

            batch_tensor = torch.stack(tensors).to(self.device)
            feat = self.features(batch_tensor)
            feat = torch.flatten(feat, 1)
            emb = self.projection(feat)
            embeddings.append(emb.cpu().numpy().astype(np.float32))

        return np.vstack(embeddings)


class ConvAutoencoder(nn.Module):
    """
    Lightweight Convolutional Autoencoder for unsupervised road surface
    reconstruction. Anomalous patterns (potholes, cracks) exhibit high
    reconstruction error compared to learned smooth road textures.
    """

    def __init__(self):
        super().__init__()
        # Encoder: 3 x 64 x 64 -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        # Decoder: 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 3 x 64 x 64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def get_reconstruction_error(self, patch_bgr: np.ndarray, device: str = "cpu") -> float:
        """
        Compute mean squared reconstruction error for a single BGR patch.
        """
        self.eval()
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            recon = self.forward(tensor)
            error = torch.mean((tensor - recon) ** 2).item()

        return float(error)

