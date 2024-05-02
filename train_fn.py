import sys

import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F


def custom_train_fn(
    disc_P, disc_M, gen_M, gen_P, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    P_reals = 0
    P_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (photo, monet) in enumerate(loop):
        photo = photo.to(config.DEVICE)
        monet = monet.to(config.DEVICE)

        # Train discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_photo = gen_P(monet)
            D_P_real = disc_P(photo)
            D_P_fake = disc_P(fake_photo.detach())
            P_reals += D_P_real.mean().item()
            P_fakes += D_P_fake.mean().item()
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss

            fake_monet = gen_M(photo)
            D_M_real = disc_M(monet)
            D_M_fake = disc_M(fake_monet.detach())
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_real_loss + D_M_fake_loss

            D_loss = (D_P_loss + D_M_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial losses
            D_P_fake = disc_P(fake_photo)
            D_M_fake = disc_M(fake_monet)
            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))
            loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))

            # cycle losses
            cycle_monet = gen_M(fake_photo)
            cycle_photo = gen_P(fake_monet)
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
            cycle_monet_loss = l1(monet, cycle_monet)
            cycle_photo_loss = l1(photo, cycle_photo)

            # total loss
            G_loss = (
                loss_G_M
                + loss_G_P
                + cycle_monet_loss * config.LAMBDA_CYCLE
                + cycle_photo_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(
                fake_photo * 0.5 + 0.5, f"{config.SAVED_IMAGE_DIR}/photo_{idx}.png"
            )
            save_image(
                fake_monet * 0.5 + 0.5, f"{config.SAVED_IMAGE_DIR}/monet_{idx}.png"
            )

        loop.set_postfix(P_real=P_reals / (idx + 1), P_fake=P_fakes / (idx + 1))
