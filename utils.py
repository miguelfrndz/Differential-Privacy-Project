import torch
import os, sys
import numpy as np
from PIL import Image
import torch.nn as nn
from typing import Callable
from torchvision.transforms import v2
from skimage.metrics import mean_squared_error, structural_similarity as ssim

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

class PDPRegularizedLoss(nn.Module):
    def __init__(self, base_loss_fn, eta, sigma, conv_pooling = False):
        """
        Differential Privacy as Implicit Regularization (PDP) loss function.
        
        From https://arxiv.org/abs/2409.17144 
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.kappa = eta ** 2 * sigma ** 2
        self.conv_pooling = conv_pooling

    def forward(self, model, inputs, targets):
        """
        Forward pass for the PDP loss function.
        ---
        Parameters:
        model : nn.Module
            The model to be trained.
        inputs : torch.Tensor
            The input data.
        targets : torch.Tensor
            The target labels.
        """
        base_loss, reg_loss = self.base_loss_fn(model(inputs).squeeze(), targets), 0.0
        for param, input in zip(model.parameters(), inputs):
            if self.conv_pooling: param = param.mean()
            if param.requires_grad:
                reg_loss += (param ** 2 * input ** 2).sum()
        return base_loss + self.kappa * reg_loss

# FIXME: This code is utter garbage :)
# --- Gradient Leakage Attack ---
# def perform_gradient_leakage_attack(model, loss_fn, original_inputs, original_labels, device_attack, config_attack, attack_metrics):
#     model.eval()
#     params = [p for p in model.parameters() if p.requires_grad]
#     original_gradients = None
#     optimizer_zero_grad = torch.optim.SGD(params, lr = 0) # Dummy optimizer to zero grads
#     optimizer_zero_grad.zero_grad()
#     if config_attack.pdp_sgd: # Handle PDP loss calculation if used
#         original_loss = loss_fn(model, original_inputs, original_labels)
#     else:
#         outputs = model(original_inputs)
#         original_loss = loss_fn(outputs.squeeze(), original_labels)
#     # Compute gradients w.r.t model parameters
#     original_gradients = torch.autograd.grad(original_loss, params, retain_graph=False, create_graph=False) # No need to create graph here
#     original_gradients = [grad.detach() for grad in original_gradients] # Detach gradients
#     # Initialize dummy data and labels randomly
#     optimizer_zero_grad.zero_grad()
#     dummy_data = torch.randn_like(original_inputs, requires_grad = True, device = device_attack)
#     dummy_labels = torch.randn_like(original_labels, requires_grad = True, device = device_attack)
#     # Use LBFGS optimizer, which often works well for GL Attacks
#     optimizer_attack = torch.optim.LBFGS([dummy_data], lr = config_attack.gradient_attack_lr)

#     # --- 3. Iterative Reconstruction ---
#     for it in range(config_attack.gradient_attack_iterations):
#         def closure():
#             optimizer_attack.zero_grad()
#             # Compute loss and gradients for dummy data
#             model.zero_grad()
#             if config_attack.pdp_sgd:
#                 dummy_loss = loss_fn(model, dummy_data, dummy_labels)
#             else:
#                 dummy_outputs = model(dummy_data)
#                 dummy_loss = loss_fn(dummy_outputs.squeeze(), dummy_labels)
#             model.zero_grad()
#             # Recalculate loss for gradient computation, ensuring graph retention if needed
#             if config_attack.pdp_sgd:
#                 temp_loss_for_grads = loss_fn(model, dummy_data, dummy_labels)
#             else:
#                 temp_outputs_for_grads = model(dummy_data)
#                 temp_loss_for_grads = loss_fn(temp_outputs_for_grads.squeeze(), dummy_labels)
#             # Compute gradients w.r.t model parameters using dummy data
#             dummy_gradients = torch.autograd.grad(temp_loss_for_grads, params, create_graph = True)
#             # Calculate the gradient matching loss (L2 distance)
#             grad_diff = 0
#             for W_orig, W_dummy in zip(original_gradients, dummy_gradients):
#                 grad_diff += ((W_orig - W_dummy)**2).sum()
#             # Backpropagate the gradient difference loss
#             alpha = config_attack.gradient_attack_alpha
#             total_attack_loss = grad_diff + alpha * dummy_loss
#             # Backpropagate the combined attack loss w.r.t. dummy_data and dummy_labels
#             total_attack_loss.backward()
#             return total_attack_loss
#         optimizer_attack.step(closure)
#         mse, ssim = calculate_attack_metrics(dummy_data, original_inputs)
#         attack_metrics['iter'].append(it)
#         attack_metrics['mse'].append(mse)
#         attack_metrics['ssim'].append(ssim)
#         print(f"Iteration {it}: MSE: {mse}, SSIM: {ssim}")
#     return dummy_data

# # --- Attack Metrics Calculation Function ---
# def calculate_attack_metrics(reconstructed_batch, original_batch):
#     """
#     Calculates MSE and SSIM between reconstructed and original image batches.
#     Args:
#         reconstructed_batch (torch.Tensor): Batch of reconstructed images (B, C, H, W).
#         original_batch (torch.Tensor): Batch of original images (B, C, H, W).
#     Returns:
#         tuple: (average_mse, average_ssim) for the batch.
#     """
#     # Move tensors to CPU and convert to NumPy arrays
#     recon_np = reconstructed_batch.detach().cpu().numpy()
#     orig_np = original_batch.detach().cpu().numpy()

#     batch_mse = 0.0
#     batch_ssim = 0.0
#     batch_size = recon_np.shape[0]
#     num_channels = recon_np.shape[1]
#     is_multichannel = num_channels > 1

#     for i in range(batch_size):
#         recon_img = recon_np[i]
#         orig_img = orig_np[i]
#         batch_mse += mean_squared_error(orig_img, recon_img)
#         # Transpose from (C, H, W) to (H, W, C) if multichannel for skimage
#         if is_multichannel:
#             orig_img_ssim = np.transpose(orig_img, (1, 2, 0))
#             recon_img_ssim = np.transpose(recon_img, (1, 2, 0))
#             # data_range assumes images are in [0, 1] range after clamping in attack
#             ssim_val = ssim(orig_img_ssim, recon_img_ssim, multichannel=True, channel_axis=-1, data_range=1.0)
#         else:
#             # For single channel (grayscale), remove channel dim: (1, H, W) -> (H, W)
#             orig_img_ssim = orig_img.squeeze(0)
#             recon_img_ssim = recon_img.squeeze(0)
#             ssim_val = ssim(orig_img_ssim, recon_img_ssim, multichannel=False, data_range=1.0)
#         batch_ssim += ssim_val
#     return batch_mse / batch_size, batch_ssim / batch_size