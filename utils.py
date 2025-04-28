import torch
import os, sys
import numpy as np
from PIL import Image
import torch.nn as nn
from typing import Callable
from torchvision.transforms import v2

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
    def __init__(self, base_loss_fn, eta, sigma):
        """
        Differential Privacy as Implicit Regularization (PDP) loss function.
        
        From https://arxiv.org/abs/2409.17144 
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.kappa = eta ** 2 * sigma ** 2

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
            if param.requires_grad:
                reg_loss += (param ** 2 * input ** 2).sum()
        return base_loss + self.kappa * reg_loss