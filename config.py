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
    max_grad_norm: float = 1.2
    epsilon: float = 25.0 # Test for 8, 25, 50
    delta: float = 1e-5 # Should be less than the inverse of the size of the training dataset

    # PDP-SGD parameters
    pdp_sgd: bool = False
    eta: float = 0.1
    sigma: float = 0.1

    # Gradient Leakage parameters
    perform_gradient_attack: bool = True # Flag to enable/disable the attack simulation
    gradient_attack_iterations: int = 300 # Number of iterations for the attack
    gradient_attack_lr: float = 1.0 # Learning rate for the attack optimizer
    gradient_attack_alpha: float = 0.2 # Regularizer ratio
