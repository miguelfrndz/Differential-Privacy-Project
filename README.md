# Differential-Privacy-Project
Repository for the Differential Privacy Project in the Information Security Course

## Project Description

This project explores the application of differential privacy techniques, such as DP-SGD or PDP-SGD (by explicit loss regularization), to mitigate privacy risks in Machine Learning models. It also evaluates the effectiveness of these techniques against gradient leakage attacks where optimization methods can be used to reconstruct training instances based on the inner gradients of a model.

## Project Structure

### Project Structure

The repository is organized as follows:

- **`train.py`**: Main script for training the models.
- **`static/`**: Directory containing plots and figures generated during training and evaluation.
- **`analysis.py`**: Jupyter Notebook with code to generate figures and perform analysis for the paper.
- **`config.py`**: Configuration file for managing project settings.
- **`data/`**: Folder containing datasets used for training and evaluation.
- **`models.py`**: Implementation of the machine learning models used in the project.
- **`utils.py`**: Auxiliary utility functions to support various tasks.
- **`gradient_attack.py`**: Code implementation for the gradient leakage attack.
- **`requirements.txt`**: File listing the dependencies required for the project.

## Project Setup

1. Clone the repository
2. Unzip the dataset and place it in the `data` folder
```bash
unzip data.zip
```
3. Create a virtual environment and install the required packages
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

## Results

## Classification Benchmarks on Differentially Private Training

| Model | $\varepsilon$ | Accuracy | Precision | Recall/Sensitivity | Specificity | $F_1$ | MCC |
|--------|---------------|----------|-----------|---------------------|-------------|-------|-----|
| **Convolutional Neural Network (Custom-CNN)** ||||||||
| Traditional Training | $\infty$ | 0.584 | 0.586 | 0.572 | 0.596 | 0.579 | 0.168 |
| DP-SGD $(\varepsilon=8)$ | 8 | 0.532 | 0.750 | 0.096 | 0.968 | 0.170 | 0.131 |
| DP-SGD $(\varepsilon=25)$ | 25 | 0.598 | 0.612 | 0.536 | 0.660 | 0.571 | 0.198 |
| DP-SGD $(\varepsilon=50)$ | 50 | 0.588 | 0.655 | 0.372 | 0.804 | 0.475 | 0.195 |
| Explicit Regularization | -- | 0.614 | 0.609 | 0.636 | 0.592 | 0.622 | 0.228 |
| **Pretrained ResNet50 (*Fine-Tuning* Only the *Classification Head*)** ||||||||
| Traditional Training | $\infty$ | 0.886 | 0.873 | 0.904 | 0.868 | 0.888 | 0.773 |
| DP-SGD $(\varepsilon=8)$ | 8 | 0.844 | 0.831 | 0.864 | 0.824 | 0.847 | 0.689 |
| DP-SGD $(\varepsilon=25)$ | 25 | 0.868 | 0.871 | 0.864 | 0.872 | 0.868 | 0.736 |
| DP-SGD $(\varepsilon=50)$ | 50 | 0.866 | 0.877 | 0.852 | 0.880 | 0.864 | 0.732 |
| Explicit Regularization | -- | 0.892 | 0.883 | 0.904 | 0.880 | 0.893 | 0.784 |
| **Pretrained DINOv2-w/ Registers (Vision Transformer, *Fine-Tuning* Only the *Classification Head*)** ||||||||
| Traditional Training | $\infty$ | 0.972 | 0.992 | 0.952 | 0.992 | 0.971 | 0.945 |
| DP-SGD $(\varepsilon=8)$ | 8 | 0.964 | 0.992 | 0.936 | 0.992 | 0.963 | 0.929 |
| DP-SGD $(\varepsilon=25)$ | 25 | 0.968 | 0.996 | 0.940 | 0.996 | 0.967 | 0.938 |
| DP-SGD $(\varepsilon=50)$ | 50 | 0.968 | 0.996 | 0.940 | 0.996 | 0.967 | 0.938 |
| Explicit Regularization | -- | 0.968 | 0.996 | 0.940 | 0.996 | 0.967 | 0.938 |

## Gradient Leakage Reconstruction Attack

1. **Structural Similarity Index Measure (SSIM) Plot**
    ![SSIM Plot](static/SSIM_Evolution.pdf)
    ![SSIM Plot Filtered](static/SSIM_Evolution_Filtered.pdf)

2. **Reconstruction Loss Plot**  
    ![Loss Plot](static/Loss_Evolution.pdf)

## Good References

## Libraries and Tools
- [Opacus Library by META](https://opacus.ai/)
- [List of Available Models in `torchvision`](https://pytorch.org/vision/main/models.html#classification)

## Papers & Lecture Notes
- [Differential Privacy: DP-SGD (Gradient Clipping and Gaussian Noise Addition)](https://arxiv.org/pdf/1607.00133) (Original Paper by Goodfellow)
- [Opacus Paper](https://arxiv.org/pdf/2109.12298)
- [Paper on Gradient Leakage Attacks](https://arxiv.org/abs/2004.10397)
- [DP via Loss Function Regularization (Univ. Granada)](https://arxiv.org/abs/2409.17144)
- [Paper on Synthetic Data Generation](https://arxiv.org/pdf/2306.01684)
- [Good Notes on Possible Attacks to DP-SGD](https://www.khoury.northeastern.edu/home/alina/classes/Fall2021/Lecture17_Notes.pdf)
- [Another Good Reference on Gradient Leakage Attacks](https://link.springer.com/article/10.1007/s10462-023-10550-z)

## Datasets
- [Hot-Dog vs Not Hot-Dog Dataset](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog/data)
- [Toxicity Classification Dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)