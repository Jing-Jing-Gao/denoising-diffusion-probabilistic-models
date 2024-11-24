import os
import argparse
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from torchvision.models import inception_v3
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

# create a simple 2D convolutional neural network. This network 
# is essentially going to try to estimate the diffusion process --- we
# can then use this network to generate realistic images.

# First, we create a single CNN block which we will stack to create the
# full network. We use `LayerNorm` for stable training and no batch dependence.
class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expected_shape,
        act=nn.GELU,
        kernel_size=7,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm(expected_shape),
            act()
        )

    def forward(self, x):
        return self.net(x)

# We then create the full CNN model, which is a stack of these blocks
# according to the `n_hidden` tuple, which specifies the number of
# channels at each hidden layer.
class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        expected_shape=(28, 28),
        n_hidden=(64, 128, 64),
        kernel_size=7,
        last_kernel_size=3,
        time_embeddings=16,
        act=nn.GELU,
    ) -> None:
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        ## This part is literally just to put the single scalar "t" into the CNN
        ## in a nice, high-dimensional way:
        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128), act(),
            nn.Linear(128, 128), act(),
            nn.Linear(128, 128), act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Shapes of input:
        #    x: (batch, chan, height, width)
        #    t: (batch,)

        embed = self.blocks[0](x)
        # ^ (batch, n_hidden[0], height, width)

        # Add information about time along the diffusion process
        #  (Providing this information by superimposing in latent space)
        embed += self.time_encoding(t)
        #         ^ (batch, n_hidden[0], 1, 1) - thus, broadcasting
        #           to the entire spatial domain

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed

# Next, we define the actual diffusion model, which specifies the training
# schedule, takes an arbitrary model for estimating the
# diffusion process (such as the CNN above),
# and computes the corresponding loss (as well as generating samples).
class DDPM(nn.Module):
    def __init__(
        self,
        gt,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt

        noise_schedule = ddpm_schedules(betas[0], betas[1], n_T)

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.beta_t  # Exists! Set by register_buffer
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])
        self.alpha_t

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Algorithm 18.1 in Prince"""

        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting

        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t. Loss is what we return.

        return self.criterion(eps, self.gt(z_t, t / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        """Algorithm 18.2 in Prince"""

        _one = torch.ones(n_sample, device=device)
        z_t = torch.randn(n_sample, *size, device=device)
        for i in range(self.n_T, 0, -1):
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(z_t, (i/self.n_T) * _one)
            z_t /= torch.sqrt(1 - beta_t)

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of the algorithm)

        return z_t

#  we define a diffusion model using gamma noise
class DDGM(nn.Module):
    def __init__(
        self,
        gt,
        betas: Tuple[float, float],
        theta_0:float,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt

        # Generating noise schedule for gamma noise
        noise_schedule = ddgm_schedules(betas[0], betas[1], theta_0, n_T)

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])
        self.register_buffer("k_t", noise_schedule["k_t"])
        self.register_buffer("theta_t", noise_schedule["theta_t"])
        self.register_buffer("k_t_bar", noise_schedule["k_t_bar"])

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Randomly sample time step
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)

        # Reshape and repeat tensors for broadcasting
        alpha_t = self.alpha_t[t].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        k_t_bar = self.k_t_bar[t].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
        theta_t = self.theta_t[t].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])

        # Sample gamma noise
        g_t_bar = torch.distributions.Gamma(k_t_bar, 1/theta_t).sample()

        # Calculate z_t
        z_t = torch.sqrt(alpha_t) * x + g_t_bar - k_t_bar * theta_t

        return self.criterion((g_t_bar - k_t_bar * theta_t) / torch.sqrt(1 - alpha_t), self.gt(z_t, t / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        _one = torch.ones(n_sample, device=device)
        theta_T = self.theta_t[self.n_T]
        k_T_bar = self.k_t_bar[self.n_T]
        init_samples_shape = (n_sample, *size)
        z = torch.distributions.Gamma(torch.full(init_samples_shape, k_T_bar), torch.full(init_samples_shape, 1 / theta_T)).sample().to(device=device)
        z_t = z - theta_T * k_T_bar
        for i in range(self.n_T, 0, -1):
            alpha_t = self.alpha_t[i]
            alpha_t_minus_1 = self.alpha_t[i-1]
            beta_t = self.beta_t[i]
            k_t_bar = self.k_t_bar[i]
            theta_t = self.theta_t[i]

            grad = self.gt(z_t, (i / self.n_T) * _one)
            x0 = (1 / alpha_t.sqrt()) * (z_t - (1 - alpha_t).sqrt() * grad)
            z_t = (alpha_t_minus_1.sqrt() * beta_t / (1 - alpha_t)) * x0 + ((1 - beta_t).sqrt() * (1 - alpha_t_minus_1) / (1 - alpha_t)) * z_t

            if i > 1:
                z = torch.distributions.Gamma(torch.full(init_samples_shape, k_t_bar), torch.full(init_samples_shape, 1 / theta_t)).sample().to(device=device)
                z = (z - theta_t * k_t_bar) / torch.sqrt(1 - alpha_t)
                z_t += torch.sqrt(beta_t) * z

        return z_t

# The following function creates a DDPM training schedule for use when evaluating
# and training the diffusion model:
def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """Returns pre-computed schedules for DDPM sampling with a linear noise schedule."""
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))  # Cumprod in log-space (better precision)

    return {"beta_t": beta_t, "alpha_t": alpha_t}


# The following function creates a DDGM training schedule for use when evaluating
# and training the designed diffusion model:
def ddgm_schedules(beta1: float, beta2: float, theta_0:float, T: int) -> Dict[str, torch.Tensor]:
    """Returns pre-computed schedules for DDGM sampling with a linear noise schedule."""
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))  # Cumprod in log-space (better precision)

    k_t = beta_t/(alpha_t*theta_0*theta_0)
    theta_t = torch.sqrt(alpha_t)*theta_0
    k_t_bar = torch.cumsum(k_t, dim=0)
    return {"beta_t": beta_t, "alpha_t":alpha_t, "k_t":k_t, "theta_t":theta_t,"k_t_bar":k_t_bar}

# Define a function to compute the activations
def compute_activations(images, model):
    activations = []
    model.eval()
      # Compute activations
    batch_size = 128
    # device = images.device

    # # Check if CUDA is available
    # if torch.cuda.is_available():
    #     # Define the device
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    # device = torch.device("cpu")  # Use CPU device
    device = torch.device("cuda")
    with torch.no_grad():
        for batch in DataLoader(images, batch_size=batch_size):
            batch = batch.to(device)
            pred = model(batch)
            activations.append(pred)
    activations = torch.cat(activations, dim=0)
    return activations

def calculate_fid_score(real_activations, generated_activations):
    # Convert activations to PyTorch tensors
    real_activations_tensor = torch.tensor(real_activations)
    generated_activations_tensor = torch.tensor(generated_activations)

    # Calculate mean and covariance matrices for real and generated activations
    mu_real = torch.mean(real_activations_tensor, dim=0)
    mu_generated = torch.mean(generated_activations_tensor, dim=0)
    sigma_real = torch_cov(real_activations_tensor, rowvar=False)
    sigma_generated = torch_cov(generated_activations_tensor, rowvar=False)

    # Calculate square root of product of covariance matrices
    sqrt_sigma_real_sigma_generated = sqrtm(sigma_real @ sigma_generated)
    
    # Ensure real-valued result if square root is complex
    if np.iscomplexobj(sqrt_sigma_real_sigma_generated):
        sqrt_sigma_real_sigma_generated = sqrt_sigma_real_sigma_generated.real

    # Calculate FID score
    fid_score = torch.norm(mu_real - mu_generated)**2 + torch.trace(sigma_real + sigma_generated - 2 * sqrt_sigma_real_sigma_generated)
    return fid_score

# Function to calculate covariance matrix
def torch_cov(m, rowvar=False):
    if m.dim() <= 1:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def  Calculate_FID_score(real_images, generated_images):
    # Load Inception-v3 model pretrained on ImageNet
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device=x.device)
    inception_model.eval()
  # Preprocess images
    preprocess = transforms.Compose([
      transforms.ToPILImage(),  # Convert tensor to PIL Image
      transforms.Grayscale(num_output_channels=3),  # Convert single channel to three channels
      transforms.Resize(299),
      transforms.CenterCrop(299),
      transforms.ToTensor(),  # Convert image to tensor
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
  ])


    real_images_preprocessed = torch.stack([preprocess(image) for image in real_images])
    generated_images_preprocessed = torch.stack([preprocess(image) for image in generated_images])

  # Compute activations
    batch_size = 128
    device = real_images.device

    real_images_preprocessed = torch.stack([preprocess(image.cpu()) for image in real_images])
    generated_images_preprocessed = torch.stack([preprocess(image.cpu()) for image in generated_images])

  # Compute activations
    real_activations = compute_activations(real_images_preprocessed, inception_model)
    generated_activations = compute_activations(generated_images_preprocessed, inception_model)

  # Move tensors to CPU and detach from computational graph
    real_activations_cpu = real_activations.cpu().detach().numpy()
    generated_activations_cpu = generated_activations.cpu().detach().numpy()

  # Calculate FID score
    fid_score = calculate_fid_score(real_activations_cpu, generated_activations_cpu)
  # print("FID Score:", fid_score.item())
    return fid_score.item()

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DDPM or DDGM.")
    parser.add_argument("param", help="DDPM or DDGM.")
    args = parser.parse_args()

    # Create directory for saving contents
    directory = f"contents_{args.param}"
    os.makedirs(directory, exist_ok=True)

    # Load dataset
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    
    # Define the model for estimator
    gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)

    # Choose model either to use Gaussian noise (DDPM) for diffusion or gamma noise (DDGM)
    if args.param == "DDGM":
        model = DDGM(gt=gt, betas=(1e-4, 0.02), theta_0=0.001, n_T=1000)
    elif args.param == "DDPM":
        model = DDPM(gt=gt, betas=(1e-4, 0.02), n_T=1000)

    # Define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
    
    # Initialize Accelerator
    accelerator = Accelerator()

    # Wrap model, optimizer, and dataloader with Accelerate to handle device placement and gradient accumulation
    model, optim, dataloader = accelerator.prepare(model, optim, dataloader)
    
    # Define number of epochs and lists for tracking losses and FID scores
    n_epoch = 18
    losses = []
    avg_losses = []
    fid_scores = []

    for i in range(n_epoch):
        model.train()  # Set model to training mode

        pbar = tqdm(dataloader)  # Wrap loop with a visual progress bar
        for x, _ in pbar:
            optim.zero_grad()  # Zero gradients

            loss = model(x)  # Forward pass

            loss.backward()  # Backward pass
            losses.append(loss.item())  # Track loss
            avg_loss = np.average(losses[min(len(losses)-100, 0):])  # Calculate running average of loss
            pbar.set_description(f"loss: {avg_loss:.3g}")  # Display running average loss in progress bar

            optim.step()  # Update weights
        
        avg_losses.append(avg_loss)  # Track average loss for each epoch

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            xh = model.sample(128, (1, 28, 28), accelerator.device)  # Generate samples
            grid = make_grid(xh, nrow=4)  # Create grid of generated samples

            # Save generated samples to directory
            save_image(grid, f"./contents_{args.param}/sample_{i:04d}.png")

            # Save model
            torch.save(model.state_dict(), f"./mnist.pth")

    # Plot the loss curve
    plt.plot(range(1, n_epoch + 1), avg_losses, marker='o', linestyle='-', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.show()

    # # Plot the fid curve
    # plt.plot(range(1, n_epoch + 1), fid_scores, marker='o', linestyle='-', markersize=3)
    # plt.xlabel('Epoch')
    # plt.ylabel('FID score')
    # plt.title('Fid Score Curve')
    # plt.grid(True)
    # plt.show()