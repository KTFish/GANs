import cv2

# TODO: Debug through this file
# TODO: Download Kaggle dataset: https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset
# TODO: Train the model


import torch
import torch.nn as nn
import torch.optim as optim
import config
from utils import save_checkpoint, load_checkpoint, save_some_examples
from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# from tqdm import tqdm


def train_fn(
    disc: Discriminator,
    gen: Generator,
    dataloader: DataLoader,
    opt_disc: optim.Optimizer,
    opt_gen: optim.Optimizer,
    l1: nn.L1Loss,
    bce: nn.BCEWithLogitsLoss,
    g_scaler: torch.cuda.amp.GradScaler,
    d_scaler: torch.cuda.amp.GradScaler,
):
    for idx, (x, y) in enumerate(dataloader):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train the Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)

            d_real = disc(x, y)
            d_fake = disc(x, y_fake.detach())

            d_real_loss = bce(d_real, torch.ones_like(d_real))
            d_fake_loss = bce(d_fake, torch.zeros_like(d_fake))
            d_loss = (
                d_real_loss + d_fake_loss
            ) / 2  # According to paper:  Devide by 2 to make discriminator learn slower

        disc.zero_grad()
        d_scaler.scale(d_loss).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            d_fake = disc(x, y_fake)
            g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            g_loss = g_fake_loss + l1(y_fake, y) * config.L1_LAMBDA

        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward(retain_graph=True)
        g_scaler.step(opt_gen)
        g_scaler.update()

        return g_loss, d_loss


def main():
    print(
        f"Pix2Pix model will be trained on {config.DEVICE} for {config.NUM_EPOCHS} epochs."
    )

    # Setup Generator and Discriminator
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )

    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)

    # Setup Dataset and Dataloaders
    train_dataset = MapDataset(root_dir=r"./architectures\pix2pix\data\maps\train")
    val_dataset = MapDataset(root_dir=r"architectures\pix2pix\data\maps\val")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        g_loss, d_loss = train_fn(
            disc,
            gen,
            train_dataloader,
            opt_disc,
            opt_gen,
            l1_loss,
            bce,
            g_scaler,
            d_scaler,
        )

        print(
            f"Epoch {epoch} | Generator Loss {g_loss:.3f} | Discriminator Loss {d_loss:.3f}"
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)

        save_some_examples(
            gen, val_dataloader, epoch, folder=r"architectures\pix2pix\evaluation"
        )


if __name__ == "__main__":
    main()
