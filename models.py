import torch
import torch.nn as nn

class DINO_wRegisters(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Load Base DINOv2 w/ Registers
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
        self.dino.eval()
        # Freeze the backbone
        for param in self.dino.parameters():
            param.requires_grad = False
        # Modify the last layer to match our number of classes
        num_ftrs = self.dino.linear_head.in_features
        self.dino.linear_head = nn.Linear(num_ftrs, 1)
        # Ensure the last layer's parameters are trainable
        for param in self.dino.linear_head.parameters():
            param.requires_grad = True
    def forward(self, x):
        # Forward pass through the model
        x = self.dino(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def MNIST_training_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 10, num_channels = 1, num_filters = 12):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, kernel_size = 5, padding = 5//2, stride = 2),
            nn.Sigmoid(),
            nn.Conv2d(num_filters, num_filters, kernel_size = 5, padding = 5//2, stride = 2),
            nn.Sigmoid(),
            nn.Conv2d(num_filters, num_filters, kernel_size = 5, padding = 5//2, stride = 1),
            nn.Sigmoid(),
            nn.Conv2d(num_filters, num_filters, kernel_size = 5, padding = 5//2, stride = 1),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(num_filters * 8 * 8, num_classes)
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out