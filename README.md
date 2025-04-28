# Differential-Privacy-Project
Repository for the Differential Privacy Project in the Information Security Course

## Project Description

TODO

## Project Structure

TODO

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

## Datasets
- [Hot-Dog vs Not Hot-Dog Dataset](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog/data)
- [Toxicity Classification Dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)