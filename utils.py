import random
import torch
import os
import numpy as np
import config


def save_checkpoint(model, optimizer, filepath: str, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpoint_file_complete_path = os.path.join(filepath, filename)

    # If path does not exists create one
    if not (os.path.exists(filepath)):
        os.mkdir(filepath)

    torch.save(checkpoint, checkpoint_file_complete_path)


def load_checkpoint(checkpoint_file, model, optimizer, filepath: str, lr):
    print("=> Loading checkpoint")
    checkpoint_file_complete_path = os.path.join(filepath, checkpoint_file)
    checkpoint = torch.load(checkpoint_file_complete_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
