"""
This is an old version of the code that was used to perform gradient leakage attacks on a simple CNN model.

Please, ignore this code. It is not used in the current version of the project.
"""
import torch
import sys, os
import numpy as np
import torch.nn as nn
from config import Config
from models import CustomCNN
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from utils import perform_gradient_leakage_attack
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

config = Config()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.device = device
print(f"Using device: {device}")

# Define transforms
transform = v2.Compose([
    v2.ToImage(), 
    v2.Resize(32),
    v2.CenterCrop(32),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST

])

# Load MNIST dataset
train_dataset = MNIST(root = "./data", train = True, download = True, transform = transform)
test_dataset = MNIST(root = "./data", train = False, download = True, transform = transform)

train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(768, 10)
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = SimpleCNN().to(device)
num_epochs = 2

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_train_loss = 0.0
    epoch_train_predictions = []
    epoch_train_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)
        predicted = torch.argmax(outputs, dim = 1)
        epoch_train_predictions.extend(predicted.cpu().numpy())
        epoch_train_labels.extend(labels.cpu().numpy())

    # --- Calculate Training Metrics ---
    epoch_train_loss = running_train_loss / len(train_dataset)
    epoch_train_accuracy = accuracy_score(epoch_train_labels, epoch_train_predictions)
    epoch_train_mcc = matthews_corrcoef(epoch_train_labels, epoch_train_predictions)
    
    print(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Train MCC: {epoch_train_mcc:.4f}")

    model.eval() # Set model to evaluation mode
    running_test_loss = 0.0
    epoch_test_predictions = []
    epoch_test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item() * inputs.size(0)
            predicted = torch.argmax(outputs, dim = 1)
            epoch_test_predictions.extend(predicted.cpu().numpy())
            epoch_test_labels.extend(labels.cpu().numpy())

    # --- Calculate Testing Metrics ---
    epoch_test_loss = running_test_loss / len(test_dataset)
    epoch_test_accuracy = accuracy_score(epoch_test_labels, epoch_test_predictions)
    epoch_test_mcc = matthews_corrcoef(epoch_test_labels, epoch_test_predictions)

    print(f"Epoch {epoch + 1}/{config.num_epochs}, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_accuracy:.4f}, Test MCC: {epoch_test_mcc:.4f}")

# --- Gradient Leakage Attack ---
if config.perform_gradient_attack:
    # Dictionary to store attack metrics over epochs
    attack_metrics = {'iter': [], 'mse': [], 'ssim': []}
    print("\nPerforming Gradient Leakage Attack...")
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs[0].unsqueeze(0).to(device), labels[0].unsqueeze(0).to(device)
    input_shape = inputs.shape
    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    true_gradients = torch.autograd.grad(loss, model.parameters())
    # true_gradients = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    true_gradients = [g.detach().clone() for g in true_gradients]
    reconstructed_inputs = perform_gradient_leakage_attack(model, true_gradients, input_shape, inputs, config, attack_metrics)
    # Save attack metrics to file
    np.savez('attack_metrics.npz', **attack_metrics)
    print("Attack metrics saved to 'attack_metrics.npz'.")

    reconstructed_inputs = np.transpose(reconstructed_inputs.squeeze(0).cpu(), (1, 2, 0))
    plt.imshow(reconstructed_inputs, cmap = 'gray')
    plt.show()