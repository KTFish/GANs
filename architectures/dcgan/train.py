import torch
from torch import nn
import config
from dataset import create_dataloders
from generator import Generator
from discriminator import Discriminator
from torch.optim import Adam
import utils
from tqdm import tqdm


def train() -> None:
    # Create DataLoaders
    train_loader, test_loader = create_dataloders()
    assert type(train_loader) == torch.utils.data.DataLoader

    # Setup Models
    gen = Generator().to(config.DEVICE)
    disc = Discriminator().to(config.DEVICE)

    # Setup Optimizers
    gen_opt = Adam(params=gen.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    disc_opt = Adam(
        params=disc.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS
    )

    # Initialize Weights
    gen = gen.apply(utils.weights_init)
    disc = disc.apply(utils.weights_init)

    # Setup Loss Function
    criterion = nn.BCEWithLogitsLoss()  # This is more efficient than nn.BCELoss

    # Training loop
    gen.train()
    disc.train()

    avg_gen_loss, avg_disc_loss = 0, 0
    curr_step, display_step = 1, 2
    print(f"DCGAN starts training on {config.DEVICE} for {config.EPOCHS} epochs.")
    for epoch in range(config.EPOCHS):
        # Train Generator
        for real, _ in tqdm(train_loader):  # TODO: Train generator function?
            curr_batch_size = len(real)
            real = real.to(config.DEVICE)

            ### Train Discriminator ### # TODO: Train discriminator function?
            disc_opt.zero_grad()

            # Get generator prediction form noise
            fake_noise = utils.sample_noise(
                curr_batch_size, config.Z_DIM, config.DEVICE
            )
            fake = gen(fake_noise)

            # Compare discriminators prediction for a batch of fake image with a tensor of zeros
            # (0 should be predicted for every fake image)
            disc_fake_pred = disc(fake.detach())
            ideal_output = torch.zeros_like(disc_fake_pred)
            disc_fake_loss = criterion(disc_fake_pred, ideal_output)

            # Compare discriminators prediction for a batch of real images with a tensor of ones
            # (for a real image a 1 should be predicted)
            disc_real_pred = disc(real)
            ideal_output = torch.ones_like(disc_real_pred)
            disc_real_loss = criterion(disc_real_pred, ideal_output)

            # Combine the discriminator loss for real and fake
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            avg_disc_loss += disc_loss.item() / curr_batch_size

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            disc_opt.step()

            ### Train Generator ###
            gen_opt.zero_grad()
            fake_noise = utils.sample_noise(
                curr_batch_size, config.Z_DIM, config.DEVICE
            )
            fake = gen(fake_noise)
            disc_fake_pred = disc(fake)

            # Compare discriminator predictions with ones
            # (the generator should be able to fool the discriminator, so for fakes it predict 1 instead of 0)
            ideal_output = torch.ones_like(disc_fake_pred)
            gen_loss = criterion(disc_fake_pred, ideal_output)

            # Keep track of generator loss
            avg_gen_loss += gen_loss.item() / curr_batch_size

            # Update gradients
            gen_loss.backward()

            # Update oprimizer
            gen_opt.step()

        ### Visualization code ###
        if curr_step % display_step == 0 and curr_step > 0:
            # Show progress
            utils.print_training_progress(epoch, curr_step, avg_gen_loss, avg_disc_loss)
            utils.show_tensor_images(fake, epoch=epoch, category="fake")
            utils.show_tensor_images(real, epoch=epoch, category="real")
            avg_gen_loss, avg_disc_loss = 0, 0
        curr_step += 1


if __name__ == "__main__":
    train()
