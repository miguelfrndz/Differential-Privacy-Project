from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = 'ResNet50' # 'ResNet50', 'CustomCNN', or 'DINO_wRegisters'
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    validation_split: float = 0.25
    early_stopping_patience: int = 20
    early_stopping_delta: float = 0.01
    device: str = None

    # Debug mode (for testing purposes)
    debug_mode: bool = False

    # DP-SGD parameters
    dp_sgd: bool = False
    max_grad_norm = 1.2
    epsilon = 50.0
    delta = 1e-5 # Should be less than the inverse of the size of the training dataset
