import torch
from dataset import PhotoMonetDataset
from discriminator import Discriminator
from generator import Generator

from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchvision.utils import save_image
import config
from utils import load_checkpoint, save_checkpoint, seed_everything


def train_fn(
    gen_P,
    disc_P,
    gen_M,
    disc_M,
    opt_gen,
    opt_disc,
    l1,
    mse,
    loader,
    g_scaler,
    d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (photo, monet) in enumerate(loop, 0):
        real_photo = photo.to(config.DEVICE)
        real_monet = monet.to(config.DEVICE)
        running_descriminator_loss, running_generator_loss = 0.0, 0.0

        # TRAIN DISCRIMINATOR FIRST disc_P, disc_M
        with torch.cuda.amp.autocast():
            # generate a fake photo
            fake_photo = gen_P(real_monet)
            disc_photo_real = disc_P(real_photo)
            disc_photo_fake = disc_P(fake_photo.detach())
            disc_photo_real_loss = mse(
                disc_photo_real, torch.ones_like(disc_photo_real)
            )
            disc_photo_fake_loss = mse(
                disc_photo_fake, torch.zeros_like(disc_photo_fake)
            )
            disc_photo_total_loss = disc_photo_real_loss + disc_photo_fake_loss

            # Generate a fake monet
            fake_monet = gen_M(real_photo)
            disc_monet_real = disc_M(real_monet)
            disc_monet_fake = disc_M(fake_monet.detach())
            disc_monet_real_loss = mse(
                disc_monet_real, torch.ones_like(disc_monet_real)
            )
            # tensor of 1's represents the target value for real image. 1 meaning real and 0 mean fake
            disc_monet_fake_loss = mse(
                disc_monet_fake, torch.zeros_like(disc_monet_fake)
            )
            disc_monet_total_loss = disc_monet_real_loss + disc_monet_fake_loss

            # Put it together
            D_loss = (disc_photo_total_loss + disc_monet_total_loss) / 2

        # Calculate running loss for discriminator
        running_descriminator_loss += D_loss.item()

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # TRAIN GENERATORreal_photoS PPP AND M
        with torch.cuda.amp.autocast():
            # adversarial loss for both generator
            disc_photo_fake = disc_P(fake_photo)
            disc_monet_fake = disc_M(fake_monet)
            gen_photo_loss = mse(disc_photo_fake, torch.ones_like(disc_photo_fake))
            gen_monet_loss = mse(disc_monet_fake, torch.ones_like(disc_monet_fake))

            # Cycle loss
            cycle_photo = gen_P(fake_monet)
            cycle_monet = gen_M(fake_photo)
            cycle_photo = F.interpolate(
                cycle_photo,
                size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                mode="bilinear",
            )
            cycle_monet = F.interpolate(
                cycle_monet,
                size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                mode="bilinear",
            )

            cycle_photo_loss = l1(real_photo, cycle_photo)
            cycle_monet_loss = l1(real_monet, cycle_monet)

            # Identity loss
            identity_photo = gen_P(real_photo)
            identity_monet = gen_M(real_monet)
            identity_photo = F.interpolate(
                identity_photo,
                size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                mode="bilinear",
            )
            identity_monet = F.interpolate(
                identity_monet,
                size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                mode="bilinear",
            )
            identity_photo_loss = l1(real_photo, identity_photo)
            identity_monet_loss = l1(real_monet, identity_monet)

            # Sum up
            G_loss = (
                cycle_photo_loss * config.LAMBDA_CYCLE
                + cycle_monet_loss * config.LAMBDA_CYCLE
                + identity_photo_loss * config.LAMBDA_IDENTITY
                + identity_monet_loss * config.LAMBDA_IDENTITY
                + gen_photo_loss
                + gen_monet_loss
            )

        # Calculate running loss for generator
        running_generator_loss += G_loss.item()

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # After every 200 batch save the images as well as print the G and D loss
        if idx % 200 == 0:
            save_image(
                fake_photo * 0.5 + 0.5, f"{config.SAVED_IMAGE_DIR}/photo_{idx}.png"
            )
            save_image(
                fake_monet * 0.5 + 0.5, f"{config.SAVED_IMAGE_DIR}/monet_{idx}.png"
            )
            print(f"Batch : {idx+1}")
            print(
                f"Generator Loss: {running_generator_loss/200:.2f} & Discriminator Loss: {running_descriminator_loss/200:.2f}"
            )
            running_descriminator_loss, running_generator_loss = 0.0, 0.0


def main(is_save: bool = True, is_load: bool = False):
    seed_everything()
    disc_P = Discriminator(input_channels=3).to(config.DEVICE)
    disc_M = Discriminator(input_channels=3).to(config.DEVICE)
    gen_P = Generator(image_channels=3, num_residuals=9).to(config.DEVICE)
    gen_M = Generator(image_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_P.parameters()) + list(gen_P.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(disc_M.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL or is_load:
        load_checkpoint(
            checkpoint_file=config.CHECKPOINT_GEN_P,
            model=gen_P,
            optimizer=opt_gen,
            filepath=config.CHECKPOINT_SAVE_DIR,
            lr=config.LEARNING_RATE,
        )
        load_checkpoint(
            checkpoint_file=config.CHECKPOINT_CRITIC_P,
            model=disc_P,
            optimizer=opt_disc,
            filepath=config.CHECKPOINT_SAVE_DIR,
            lr=config.LEARNING_RATE,
        )
        load_checkpoint(
            checkpoint_file=config.CHECKPOINT_GEN_M,
            model=gen_M,
            optimizer=opt_gen,
            filepath=config.CHECKPOINT_SAVE_DIR,
            lr=config.LEARNING_RATE,
        )
        load_checkpoint(
            checkpoint_file=config.CHECKPOINT_CRITIC_M,
            model=disc_M,
            optimizer=opt_disc,
            filepath=config.CHECKPOINT_SAVE_DIR,
            lr=config.LEARNING_RATE,
        )

    dataset = PhotoMonetDataset(
        root_dir_photo=config.TRAIN_DIR + "/photo_jpg/",
        root_dir_monet=config.TRAIN_DIR + "/monet_jpg/",
        transform=config.transforms,
    )
    loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    # train float16 instead of float32
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch: {epoch}")

        train_fn(
            gen_P,
            disc_P,
            gen_M,
            disc_M,
            opt_gen,
            opt_disc,
            L1,
            mse,
            loader,
            g_scaler,
            d_scaler,
        )
        if config.SAVE_MODEL or is_save:
            save_checkpoint(
                model=gen_P,
                optimizer=opt_gen,
                filepath=config.CHECKPOINT_SAVE_DIR,
                filename=config.CHECKPOINT_GEN_P,
            )
            save_checkpoint(
                model=disc_P,
                optimizer=opt_disc,
                filepath=config.CHECKPOINT_SAVE_DIR,
                filename=config.CHECKPOINT_CRITIC_P,
            )
            save_checkpoint(
                model=gen_M,
                optimizer=opt_gen,
                filepath=config.CHECKPOINT_SAVE_DIR,
                filename=config.CHECKPOINT_GEN_M,
            )
            save_checkpoint(
                model=disc_M,
                optimizer=opt_disc,
                filepath=config.CHECKPOINT_SAVE_DIR,
                filename=config.CHECKPOINT_CRITIC_M,
            )


if __name__ == "__main__":
    main(is_load=False, is_save=False)
