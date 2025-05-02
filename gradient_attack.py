"""
Deep Leakage from Gradients (DLG) Attack. Code inspired by:
https://github.com/joshtowell/deep-leakage-from-gradients
"""

import torch
import numpy as np
import sys, os, csv
import torch.nn as nn
from config import Config
import matplotlib.pyplot as plt
import scienceplots
from opacus import PrivacyEngine
from utils import cross_entropy_for_onehot, label_to_onehot, perform_gradient_leakage_attack, PDPRegularizedLoss
from torchvision.transforms import v2
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from models import SimpleCNN, MNIST_training_init
torch.manual_seed(50)

config = Config()
assert not(config.dp_sgd and config.pdp_sgd), "DP-SGD and PDP-SGD cannot be used together."
plt.style.use('science') # For "publication-like" plots
output_extension_name = "normal" if not config.dp_sgd else "DP_SGD"
output_extension_name = "PDP_SGD" if config.pdp_sgd else output_extension_name

transform = v2.Compose([
    v2.Resize(32),
    v2.CenterCrop(32),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
])

img_transform = v2.ToPILImage()

train_dataset = MNIST(root = "./data", train = True, download = True, transform = transform)
test_dataset = MNIST(root = "./data", train = False, download = True, transform = transform)

train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False)

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

if config.pdp_sgd:
    print("Using PDP regularization...")
    # Initialize the PDP loss function
    criterion = PDPRegularizedLoss(criterion, config.eta, config.sigma, conv_pooling = True)

# Compute original gradient using model
out = model(gt_data)
if config.pdp_sgd:
    y = criterion(model, gt_data, gt_onehot_label)
else:
    y = criterion(out, gt_onehot_label)
dy_dx = torch.autograd.grad(y, model.parameters())
true_gradients = list((_.detach().clone() for _ in dy_dx))

if config.dp_sgd:
    print("Using DP-SGD...")
    # Clip the gradient norm
    for i in range(len(true_gradients)):
        true_gradients[i] = true_gradients[i] / torch.max(torch.norm(true_gradients[i], p = 2)/config.max_grad_norm, torch.tensor(1.0).to(device))
        # Add noise to the gradients
        noise = torch.normal(mean = 0, std = config.epsilon * config.max_grad_norm, size = true_gradients[i].size()).to(device)
        true_gradients[i] += noise

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
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(history['reconstructed_inputs'][i * 10], cmap = 'gray')
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')
    plt.savefig(f"static/reconstructed_images_{output_extension_name}.pdf", bbox_inches = 'tight', dpi = 300)
plt.show()

plt.imshow(history['reconstructed_inputs'][-1], cmap = 'gray')
plt.axis('off')
plt.title("Reconstructed")
plt.savefig(f"static/reconstructed_image_{output_extension_name}.pdf", bbox_inches = 'tight', dpi = 300)
plt.show()

# Save history to a CSV file
csv_file_path = f"static/leakage_attack_{output_extension_name}.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration', 'Loss', 'SSIM', 'Reconstructed_Label'])
    for i in range(len(history['iteration'])):
        writer.writerow([
            history['iteration'][i],
            history['loss'][i],
            history['SSIM'][i],
            history['reconstructed_labels'][i].item() if isinstance(history['reconstructed_labels'][i], torch.Tensor) else history['reconstructed_labels'][i]
        ])

print(f"History saved to {csv_file_path}")