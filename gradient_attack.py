"""
Deep Leakage from Gradients (DLG) Attack. Code inspired by:
https://github.com/joshtowell/deep-leakage-from-gradients
"""

import torch
import sys, os
import numpy as np
import torch.nn as nn
from config import Config
import matplotlib.pyplot as plt
from utils import cross_entropy_for_onehot, label_to_onehot, perform_gradient_leakage_attack
from torchvision.transforms import v2
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from models import SimpleCNN, MNIST_training_init
torch.manual_seed(50)

config = Config()

transform = v2.Compose([
    v2.Resize(32),
    v2.CenterCrop(32),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    # v2.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
])

train_dataset = MNIST(root = "./data", train = True, download = True, transform = transform)
test_dataset = MNIST(root = "./data", train = False, download = True, transform = transform)

train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False)

img_transform = v2.ToPILImage()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.device = device
print(f"Using device: {device}")

model = SimpleCNN().to(device)
model.apply(MNIST_training_init)
criterion = cross_entropy_for_onehot

# --- Gradient Leakage Attack ---
img_index = 45
# Transformed and rescaled images
gt_data = train_dataset[img_index][0].to(device)
gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([train_dataset[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label, num_classes = 10, device = device)

print(f"Ground-Truth label is {gt_label.item()}")
print(f"Onehot label is {torch.argmax(gt_onehot_label, dim=-1).item()}")

# Compute original gradient using model
out = model(gt_data)
y = criterion(out, gt_onehot_label)
dy_dx = torch.autograd.grad(y, model.parameters())
true_gradients = list((_.detach().clone() for _ in dy_dx))

history = {
    'iteration': [],
    'loss': [],
    'SSIM': [],
    'reconstructed_inputs': [],
    'reconstructed_labels': []
}

perform_gradient_leakage_attack(model, true_gradients, input_shape = gt_data.size(), 
        output_shape = gt_onehot_label.size(), criterion = criterion, 
        original_inputs = gt_data, config = config, history = history
)

history['reconstructed_inputs'] = [img_transform(img) for img in history['reconstructed_inputs']]

# Plot generated image from every 10th iteration
plt.figure(figsize=(12, 3))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history['reconstructed_inputs'][i * 10], cmap = 'gray')
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')
plt.show()