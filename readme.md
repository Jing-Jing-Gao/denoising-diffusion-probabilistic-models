# Coursework for Medical Image 

## Description

This coursework involves the implementation and training of diffusion models for image denoising. It consists of two main parts:

- Training a Regular Denoising Diffusion Probabilistic Model (DDPM) on MNIST.
- Designing a Custom Degradation Strategy for Images inspired by the Denoising Diffusion Gamma Models (DDGM) paper by Eliya Nachmani, Robin San Roman, and Lior Wolf. The adapted DDGM approach is combined with a custom CNN model and applied to the MNIST dataset.

## Dataset
The MNIST dataset is widely used for training and testing machine learning models for image classification tasks. It consists of grayscale images of handwritten digits (0-9) with corresponding labels.

## Utility

### Create Conda Environment:
```bash
$ conda env create -f environment.yml -n your_environment_name
```
### Activate the Conda Environment: 
```bash
$ conda activate your_environment_name
```
### Coursework Section

To train the Denoising Diffusion Probabilistic Model (DDPM) and Denoising Diffusion Gamma Model (DDGM), follow the instructions below:

#### Training DDPM

To train DDPM, run the following command:

```bash
$ python main.py DDPM
```
This command will initiate the training process for DDPM. It will generate a folder named "contents_DDPM" containing the generated samples from each epoch. Additionally, it will produce the loss curves. Please note that training DDPM is time-consuming, taking nearly 2 hours to run 100 epochs with default hyperparameters.

##### Default Hyperparameters:

- CNN architecture:
- Input channels: 1
- Expected shape: (28, 28)
- Hidden layers: (16, 32, 32, 16)
- Activation function: GELU
- Betas: (1e-4, 0.02)
- Time steps (n_T): 1000
- Learning rate: 2e-4

#### Training DDGM

To train DDGM, run the following command:

```bash
$ python main.py DDGM
```
Similar to DDPM, this command will initiate the training process for DDGM. It will produce a folder named "contents_DDGM" containing the generated samples from each epoch and the corresponding loss curves. Training DDGM also takes nearly 2 hours to run 100 epochs with default hyperparameters.

##### Default Hyperparameters:

- CNN architecture:
- Input channels: 1
- Expected shape: (28, 28)
- Hidden layers: (16, 32, 32, 16)
- Activation function: GELU
- Betas: (1e-4, 0.02)
- Theta_0: 0.001
- Time steps (n_T): 1000
- Learning rate: 2e-4

#### Jupyter Notebook

Additionally, I have provided a Jupyter Notebook named `main.ipynb` for easy execution. You can run this notebook in Google Colab using a T4 GPU for faster processing.

## License
This project is licensed under the [MIT License](LICENSE).