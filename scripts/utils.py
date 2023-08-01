import torch


def get_random_noise(n_samples: int, z_dim: int, device: str = "cpu") -> torch.Tensor:
    """Returns a noise vector z (used by the generator to create an image).

    Args:
        n_samples (int): Number of samples that will be generated from that vector. Usually set to batch size.
        z_dim (int): Dimension of the noise vector.
        device (str, optional): Device on which the vector will be stored. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Noise.
    """
    return torch.randn(n_samples, z_dim, device=device)
