import random
import torch
import os
import numpy as np
import config


def save_checkpoint(model, optimizer, filepath: str, filename="my_checkpoint.pth.tar"):
    """
    Takes mode, optimizer and complete path of the folder as well as file name and saves the model weight as well as optimizer weight to that path
    PARAMS:
      model (torch model): It is pytorch model which you finished training and want to save
      optimizer (torch optimizer): It is a pytorch optimizer which holds the paramter weights for current model
      filepath(str): It is the path of the directory where you wish to save your weights.
                     It is taken from  CHECKPOINT_SAVE_DIR = "/content/drive/MyDrive/Academics/CV/saved_checkpoints" in config.py file
      filename(str): Actual name of the checkpoint which is tar.gz file

    RETURNS:
      None
    """
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
    """
    Takes mode, optimizer and complete path of the folder as well as file name and loads the saved model in the passed model parameter
    PARAMS:
      model (torch model): It is pytorch model to which you want to insert pretrained parameters
      optimizer (torch optimizer): It is a pytorch optimizer to which you want to add pretrained gradients
      filepath(str): It is the path of the directory from where you wish to access your saved checkpoint.
                     It is taken from  CHECKPOINT_SAVE_DIR = "/content/drive/MyDrive/Academics/CV/saved_checkpoints" in config.py file
      filename(str): Actual name of the checkpoint which is tar.gz file

    RETURNS:
      None
    """
    print("=> Loading checkpoint")
    checkpoint_file_complete_path = os.path.join(filepath, checkpoint_file)
    checkpoint = torch.load(checkpoint_file_complete_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed: int=42):
    """
    This function seeds everything as name suggests
    PARAMS:
    seed(int): number to seed to
    
    RETURNS:
    none
    """ 
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
