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
    
# --- Gradient Leakage Attack Simulation ---
def perform_gradient_leakage_attack(model, true_gradients, input_shape, original_inputs, config, attack_metrics):
    device = config.device
    reconstructed_inputs = torch.randn(input_shape, requires_grad = True, device = device)
    assert reconstructed_inputs.shape == original_inputs.shape, f"Reconstructed inputs shape ({reconstructed_inputs.shape}) must match original inputs shape ({original_inputs.shape})"
    label_shape = (input_shape[0], 10)
    reconstructed_labels = torch.randn(label_shape, requires_grad = True, device = device)
    # reconstructed_labels = torch.argmin(torch.sum(true_gradients[-2], dim = -1), dim = -1).detach().unsqueeze(0).requires_grad_(False)
    optimizer = torch.optim.LBFGS([reconstructed_inputs, reconstructed_labels], lr = config.gradient_attack_lr)
    # params = [p for p in model.parameters() if p.requires_grad]
    print(reconstructed_inputs.shape, reconstructed_labels.shape)
    for it in range(config.gradient_attack_iterations):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()
            outputs = model(reconstructed_inputs)
            loss = torch.nn.functional.cross_entropy(outputs, reconstructed_labels)
            dummy_grad = torch.autograd.grad(loss, model.parameters(), create_graph = True)
            # for i, g in enumerate(dummy_grad):
            #     print(f"Dummy grad {i}: ||g||={g.norm().item():.4f}")
            # for i, g in enumerate(true_gradients):
            #     print(f"True grad {i}: ||g||={g.norm().item():.4f}")
            grad_diff = sum(((dg - tg)**2).sum() / tg.numel()  for dg, tg in zip(dummy_grad, true_gradients))
            grad_diff += config.gradient_attack_alpha * loss
            grad_diff.backward()
            return grad_diff
        optimizer.step(closure)
        # with torch.no_grad():
        #     reconstructed_inputs.clamp_(-1, 1)
        mse, ssim = calculate_attack_metrics(reconstructed_inputs, original_inputs)
        attack_metrics['iter'].append(it)
        attack_metrics['mse'].append(mse)
        attack_metrics['ssim'].append(ssim)
        print(f"Iteration {it + 1}/{config.gradient_attack_iterations}: MSE: {mse}, SSIM: {ssim}")
    return reconstructed_inputs.detach()

# --- Attack Metrics Calculation Function ---
def calculate_attack_metrics(reconstructed_batch, original_batch):
    """
    Calculates MSE and SSIM between reconstructed and original image batches.
    Args:
        reconstructed_batch (torch.Tensor): Batch of reconstructed images (B, C, H, W).
        original_batch (torch.Tensor): Batch of original images (B, C, H, W).
    Returns:
        tuple: (average_mse, average_ssim) for the batch.
    """
    # Move tensors to CPU and convert to NumPy arrays
    recon_np = reconstructed_batch.detach().cpu().numpy()
    orig_np = original_batch.detach().cpu().numpy()

    batch_mse = 0.0
    batch_ssim = 0.0
    batch_size = recon_np.shape[0]
    num_channels = recon_np.shape[1]
    is_multichannel = num_channels > 1

    for i in range(batch_size):
        recon_img = recon_np[i]
        orig_img = orig_np[i]
        batch_mse += mean_squared_error(orig_img, recon_img)
        # Transpose from (C, H, W) to (H, W, C) if multichannel for skimage
        if is_multichannel:
            orig_img_ssim = np.transpose(orig_img, (1, 2, 0))
            recon_img_ssim = np.transpose(recon_img, (1, 2, 0))
            # data_range assumes images are in [0, 1] range after clamping in attack
            ssim_val = ssim(orig_img_ssim, recon_img_ssim, multichannel=True, channel_axis=-1, data_range=1.0)
        else:
            # For single channel (grayscale), remove channel dim: (1, H, W) -> (H, W)
            orig_img_ssim = orig_img.squeeze(0)
            recon_img_ssim = recon_img.squeeze(0)
            ssim_val = ssim(orig_img_ssim, recon_img_ssim, multichannel=False, data_range=1.0)
        batch_ssim += ssim_val
    return batch_mse / batch_size, batch_ssim / batch_size