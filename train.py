import torch
import os, sys
import numpy as np
import torch.nn as nn
from config import Config
import matplotlib.pyplot as plt
from opacus import PrivacyEngine
from models import CustomCNN, DINO_wRegisters
from opacus.validators import ModuleValidator
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, TensorDataset, Subset
from utils import load_data, PDPRegularizedLoss, perform_gradient_leakage_attack
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

config = Config()
assert not(config.dp_sgd and config.pdp_sgd), "DP-SGD and PDP-SGD cannot be used together."

if config.model_name == 'ResNet50':
    # Load best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights = weights)
    model_transform = weights.transforms() # Preprocessing function for ResNet50
    # Freeze all layers except the last classification layer
    for param in model.parameters():
        param.requires_grad = False
    # Modify the last layer to match our number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    # Ensure the last layer's parameters are trainable
    for param in model.fc.parameters():
        param.requires_grad = True
elif config.model_name == 'CustomCNN':
    # Load custom CNN model
    model = CustomCNN()
    model_transform = None
elif config.model_name == 'DINO_wRegisters':
    # Load DINO model with registers
    model = DINO_wRegisters()
    model_transform = None

if config.pdp_sgd:
    print("Using PDP regularization for training...")
    # Initialize the PDP loss function
    base_loss_fn = nn.BCEWithLogitsLoss()
    # base_loss_fn = nn.CrossEntropyLoss()
    conv_pooling = True if config.model_name == 'CustomCNN' else False
    criterion = PDPRegularizedLoss(base_loss_fn, config.eta, config.sigma, conv_pooling)
else:
    print("Using standard (Binary) Cross-Entropy loss for training...")
    # Use standard BCE loss function
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()

# Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

if config.dp_sgd:
    print("Using Opacus DP-SGD for training...")
    errors = ModuleValidator.validate(model, strict = False)
    if errors: 
        if config.debug_mode: print(f"Model validation errors: {errors}")
        # Fix the incompatible modules in the model (such as BatchNorm layers)
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict = False)

# Get device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
config.device = device
model.to(device)

# Load data
train_images, train_labels = load_data('train', shuffle = True, transform = model_transform)
test_images, test_labels = load_data('test', shuffle = False, transform = model_transform)

# Create indices for train and validation split
num_train = len(train_images)
indices = np.arange(num_train)
np.random.shuffle(indices)
split = int(np.floor(config.validation_split * num_train))
train_indices, val_indices = indices[split:], indices[:split]

# Create Datasets (including validation set for early stopping)
train_dataset = Subset(TensorDataset(train_images, train_labels), train_indices)
val_dataset = Subset(TensorDataset(train_images, train_labels), val_indices)
test_dataset = TensorDataset(test_images, test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = config.batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False)

# Early Stopping Parameters
early_stopping_patience = config.early_stopping_patience
early_stopping_delta = config.early_stopping_delta
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

# Training Loop
num_epochs = config.num_epochs
print(f"\nStarting training w/ Model {config.model_name}...")

if config.dp_sgd:
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module = model,
        optimizer = optimizer,
        data_loader = train_loader,
        epochs = config.num_epochs,
        target_epsilon = config.epsilon,
        target_delta = config.delta,
        max_grad_norm = config.max_grad_norm,
    )

    print(f"Using sigma={optimizer.noise_multiplier} and C={config.max_grad_norm}")

for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_train_loss = 0.0
    epoch_train_predictions = []
    epoch_train_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if config.pdp_sgd:
            loss = criterion(model, inputs, labels)
        else:
            loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)
        predicted = torch.sigmoid(outputs).squeeze() > 0.5
        # predicted = torch.argmax(outputs, dim = 1)
        epoch_train_predictions.extend(predicted.cpu().numpy())
        epoch_train_labels.extend(labels.cpu().numpy())

    # --- Calculate Training Metrics ---
    epoch_train_loss = running_train_loss / len(train_dataset)
    epoch_train_accuracy = accuracy_score(epoch_train_labels, epoch_train_predictions)
    epoch_train_mcc = matthews_corrcoef(epoch_train_labels, epoch_train_predictions)

    # --- Validation Phase ---
    model.eval() # Set model to evaluation mode
    running_val_loss = 0.0
    epoch_val_predictions = []
    epoch_val_labels = []
    # Disable gradient computation during evaluation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if config.pdp_sgd:
                loss = criterion(model, inputs, labels)
            else:
                loss = criterion(outputs.squeeze(), labels)
            running_val_loss += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs).squeeze() > 0.5
            # predicted = torch.argmax(outputs, dim = 1)
            epoch_val_predictions.extend(predicted.cpu().numpy())
            epoch_val_labels.extend(labels.cpu().numpy())

    epoch_val_loss = running_val_loss / len(val_dataset)
    epoch_val_accuracy = accuracy_score(epoch_val_labels, epoch_val_predictions)
    epoch_val_mcc = matthews_corrcoef(epoch_val_labels, epoch_val_predictions)

    if config.dp_sgd:
        epsilon = privacy_engine.get_epsilon(config.delta)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, Train MCC: {epoch_train_mcc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}, Val MCC: {epoch_val_mcc:.4f}, Epsilon: {epsilon:.2f}")
    else:
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, Train MCC: {epoch_train_mcc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}, Val MCC: {epoch_val_mcc:.4f}")

    # Early Stopping Regularization
    if epoch_val_loss < best_val_loss - early_stopping_delta:
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
            if config.pdp_sgd:
                loss = criterion(model, inputs, labels)
            else:
                loss = criterion(outputs.squeeze(), labels)
            running_test_loss += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs).squeeze() > 0.5
            # predicted = torch.argmax(outputs, dim = 1)
            epoch_test_predictions.extend(predicted.cpu().numpy())
            epoch_test_labels.extend(labels.cpu().numpy())

    final_test_loss = running_test_loss / len(test_dataset)
    final_test_accuracy = accuracy_score(epoch_test_labels, epoch_test_predictions)
    final_test_precision = precision_score(epoch_test_labels, epoch_test_predictions)
    final_test_recall = recall_score(epoch_test_labels, epoch_test_predictions)
    final_test_specificity = recall_score(epoch_test_labels, epoch_test_predictions, pos_label = 0)
    final_test_f1 = f1_score(epoch_test_labels, epoch_test_predictions)
    final_test_mcc = matthews_corrcoef(epoch_test_labels, epoch_test_predictions)

    print(f"\nFinal Test Results (Best Model):")
    print(f"\t- Loss: {final_test_loss:.4f}")
    if config.dp_sgd:
        epsilon = privacy_engine.get_epsilon(config.delta)
        print(f"\t- Epsilon: {epsilon:.2f}")
    print(f"\t- Accuracy: {final_test_accuracy:.4f}")
    print(f"\t- Precision: {final_test_precision:.4f}")
    print(f"\t- Recall/Sensitivity: {final_test_recall:.4f}")
    print(f"\t- Specificity: {final_test_specificity:.4f}")
    print(f"\t- F1 Score: {final_test_f1:.4f}")
    print(f"\t- MCC: {final_test_mcc:.4f}")

    # --- Gradient Leakage Attack ---
    # if config.perform_gradient_attack:
    #     # Dictionary to store attack metrics over epochs
    #     attack_metrics = {'iter': [], 'mse': [], 'ssim': []}
    #     print("\nPerforming Gradient Leakage Attack...")
    #     # Perform the attack on the test set
    #     inputs, labels = test_dataset.tensors
    #     # Need to ensure first dimension corresponds to batch size
    #     # inputs, labels = inputs[0].unsqueeze(0), labels[0].unsqueeze(0).unsqueeze(0)
    #     inputs, labels = inputs[0].unsqueeze(0), labels[0]
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     input_shape = inputs.shape
    #     # Unfreeze all parameters in the model
    #     for param in model.parameters():
    #         param.requires_grad = True
    #     model.eval()
    #     model.zero_grad()
    #     outputs = model(inputs)
    #     if config.pdp_sgd:
    #         loss = criterion(model, inputs, labels)
    #     else:
    #         # loss = criterion(outputs, labels)
    #         loss = criterion(outputs.squeeze(), labels)
    #     loss.backward()
    #     true_gradients = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    #     model.zero_grad()
    #     reconstructed_inputs = perform_gradient_leakage_attack(model, true_gradients, input_shape, inputs, config, attack_metrics)
    #     # Save attack metrics to file
    #     np.savez('attack_metrics.npz', **attack_metrics)
    #     print("Attack metrics saved to 'attack_metrics.npz'.")

    #     reconstructed_inputs = np.transpose(reconstructed_inputs.squeeze(0).cpu(), (1, 2, 0))
    #     plt.imshow(reconstructed_inputs)
    #     plt.show()
