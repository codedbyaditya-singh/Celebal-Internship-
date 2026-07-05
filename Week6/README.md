# Week 6: Image Denoising using Autoencoder (MNIST)

## Overview
This project implements a **Denoising Autoencoder** using the **MNIST** dataset to reconstruct clean handwritten digit images from artificially corrupted (noisy) images.

## Implementations
- Loaded and preprocessed the MNIST dataset.
- Normalized and reshaped images for model training.
- Added Gaussian noise to create noisy input images.
- Built a Convolutional Denoising Autoencoder (Encoder + Decoder).
- Trained the model using noisy images as input and clean images as target.
- Evaluated training performance using training and validation loss.
- Generated denoised images from the noisy test dataset.
- Visualized Original, Noisy, and Denoised image comparisons.
- Documented observations and final conclusions.
