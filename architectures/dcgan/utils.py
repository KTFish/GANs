import os
import torch
import config
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from typing import Tuple


def show_tensor_images(
    image: torch.Tensor,
    num_images: int = 25,
    size: Tuple[int, int, int] = (1, 28, 28),
    save_as_png: bool = True,
    epoch: int = None,
    category: str = "real",
) -> None:  # TODO: test!~!!!
    """Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.

    Args:
        image (torch.Tensor): Tensor of images.
        num_images (int, optional): Number of images to display. Defaults to 25.
        size (Tuple[int, int, int], optional): Dimensions of a single image. Defaults to (1, 28, 28).
    """
    image = (image + 1) / 2
    image_unflat = image.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis(False)
    # plt.show()

    if save_as_png:
        path = f"./data/mnist/evaluation/{category}"
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f"{path}/{epoch}.png")


def print_training_progress(
    epoch: int, step: int, gen_loss: float, disc_loss: float
) -> None:
    """Prints training progress given data.

    Args:
        epoch (int): _description_
        gen_loss (float): _description_
        disc_loss (float): _description_
    """
    print(
        f"Epoch: {epoch} | Step: {step} |Generator Loss: {gen_loss:.3f} | Discriminator Loss: {disc_loss:.3f}"
    )


def sample_noise(n_samples: int, z_dim: int, device: str = "cpu") -> torch.Tensor:
    """Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the uniform distribution.

    Args:
        n_samples (int): Number of samples that should be generated.
        z_dim (int): Lenght of the Z vector.
        device (str, optional): Device of the tensor. Defaults to "cpu".

    Returns:
        torch.Tensor: Noise vector of dimensions (n_samples, z_dim)
    """
    return torch.rand(n_samples, z_dim, device=device)


def weights_init(m):
    """Weights initialization to the normal distribution with mean 0 and standard deviation 0.02 like in dcgan paper.
    Args:
         m (nn.Module): A PyTorch module for which weights will be initialized.

    Example usage:
        gen = gen.apply(weights_init)  # Initializing weights in generator
        disc = disc.apply(weights_init)  # Initializing weights in discriminator

    Note:
        This function is typically used with the `.apply()` method of a PyTorch module to apply weight
        initialization to all relevant layers within the module.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# TODO: Save generator outputs as png fucntion


def save_model_checkpoint(
    model: torch.nn.Module, name: str, path: str = "./models/dcgan/"
) -> None:
    path = f"{path}/{name}.pth"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    print(f"Saving {name} to {path}.")
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    image = torch.randn(32, 1, 28, 28)
    show_tensor_images(image, epoch=-1, category="fake")
