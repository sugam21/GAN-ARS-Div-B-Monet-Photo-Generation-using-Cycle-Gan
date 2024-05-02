import torch
from dataset import PhotoMonetDataset
from discriminator import Discriminator
from generator import Generator
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from train_fn import custom_train_fn
import config
from utils import load_checkpoint, save_checkpoint, seed_everything

def main(is_save: bool = True, is_load: bool = False):
    seed_everything()
    disc_P = Discriminator(input_channels=3).to(config.DEVICE)
    disc_M = Discriminator(input_channels=3).to(config.DEVICE)
    gen_P = Generator(image_channels=3, num_residuals=9).to(config.DEVICE)
    gen_M = Generator(image_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_P.parameters()) + list(disc_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_M.parameters()) + list(gen_P.parameters()),
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

        custom_train_fn(
            disc_P=disc_P,
            disc_M=disc_M,
            gen_M=gen_M,
            gen_P=gen_P,
            loader=loader,
            opt_disc=opt_disc,
            opt_gen=opt_gen,
            l1=L1,
            mse=mse,
            d_scaler=d_scaler,
            g_scaler=g_scaler,
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
