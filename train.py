import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import os
from PIL import Image
from typing import Callable
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from torch.utils.data import Subset

def load_data(split : str, shuffle : bool = False, transform : Callable = None) -> torch.Tensor:
    """
    Image loader function that loads the data.
    ---
    Parameters:
    split : str
        The split of the data to load (e.g., 'train', 'test').

    shuffle : bool
        Whether to shuffle the data or not. Default is False.

    transform : Callable
        A callable transform to apply to the images (e.g., a torchvision transform function). 
        Default is None. If None, a default transform is applied that resizes the image to 
        (224, 224), converts it to a tensor, and normalizes it.
    ---    
    Returns:
    torch.Tensor
        The loaded data as a PyTorch tensor.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    data_path = os.path.join(data_dir, f'{split}/')

    if transform is None:
        # Initialize transform
        transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale = True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    images = []
    labels = []

    for label in ['hot_dog', 'not_hot_dog']:
        label_dir = os.path.join(data_path, label)
        image_files = [f for f in os.listdir(label_dir)]
        for img_file in image_files:
            labels.append(1 if label == 'hot_dog' else 0)
            img_path = os.path.join(label_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
    
    if shuffle:
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]

    return torch.stack(images), torch.tensor(labels, dtype=torch.float32)

# Load best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights = weights)

# Attention: We need to modify the last layer to match our number of classes!
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# Get device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Load data
train_images, train_labels = load_data('train', shuffle = True, transform = weights.transforms())
test_images, test_labels = load_data('test', shuffle = False, transform = weights.transforms())

# Create Tensor Dataset and DataLoaders
batch_size = 32
validation_split = 0.2

# Create indices for train and validation split
num_train = len(train_images)
indices = np.arange(num_train)
np.random.shuffle(indices)
split = int(np.floor(validation_split * num_train))
train_indices, val_indices = indices[split:], indices[:split]

# Create Subsets for train and validation
train_dataset = Subset(TensorDataset(train_images, train_labels), train_indices)
val_dataset = Subset(TensorDataset(train_images, train_labels), val_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# Loss function and optimizer
learning_rate = 1e-3
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Early Stopping Parameters
early_stopping_patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

# Training Loop
num_epochs = 100
print("\nStarting training...")

for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_train_loss = 0.0
    epoch_train_predictions = []
    epoch_train_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)
        predicted = torch.sigmoid(outputs).squeeze() > 0.5
        epoch_train_predictions.extend(predicted.cpu().numpy())
        epoch_train_labels.extend(labels.cpu().numpy())

    epoch_train_loss = running_train_loss / len(train_dataset)
    epoch_train_accuracy = accuracy_score(epoch_train_labels, epoch_train_predictions)
    epoch_train_mcc = matthews_corrcoef(epoch_train_labels, epoch_train_predictions)

    model.eval() # Set model to evaluation mode
    running_val_loss = 0.0
    epoch_val_predictions = []
    epoch_val_labels = []
    # Disable gradient computation during evaluation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            running_val_loss += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs).squeeze() > 0.5
            epoch_val_predictions.extend(predicted.cpu().numpy())
            epoch_val_labels.extend(labels.cpu().numpy())

    epoch_val_loss = running_val_loss / len(val_dataset)
    epoch_val_accuracy = accuracy_score(epoch_val_labels, epoch_val_predictions)
    epoch_val_mcc = matthews_corrcoef(epoch_val_labels, epoch_val_predictions)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, Train MCC: {epoch_train_mcc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}, Val MCC: {epoch_val_mcc:.4f}")

    # Early Stopping Regularization
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        # Save the best model state
        best_model_state = model.state_dict()
        print(f"\tValidation loss improved. Saving model state. Best Validation loss: {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(f"\tValidation loss did not improve. Patience: {epochs_no_improve}/{early_stopping_patience}")

    if epochs_no_improve >= early_stopping_patience:
        print(f"-> Early stopping triggered at epoch {epoch + 1 - early_stopping_patience}.")
        break

print("Training finished.")

# Load the best model state and perform final evaluation on the test set
if best_model_state is not None:
    print("Loading best model state...")
    model.load_state_dict(best_model_state)
    print("Best model loaded.")
    print("\nEvaluating the best model on the test set...")
    model.eval() # Set model to evaluation mode
    running_test_loss = 0.0
    epoch_test_predictions = []
    epoch_test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            running_test_loss += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs).squeeze() > 0.5
            epoch_test_predictions.extend(predicted.cpu().numpy())
            epoch_test_labels.extend(labels.cpu().numpy())

    final_test_loss = running_test_loss / len(test_dataset)
    final_test_accuracy = accuracy_score(epoch_test_labels, epoch_test_predictions)
    final_test_mcc = matthews_corrcoef(epoch_test_labels, epoch_test_predictions)

    print(f"\nFinal Test Results (Best Model):")
    print(f"\t- Loss: {final_test_loss:.4f}")
    print(f"\t- Accuracy: {final_test_accuracy:.4f}")
    print(f"\t- MCC: {final_test_mcc:.4f}")
