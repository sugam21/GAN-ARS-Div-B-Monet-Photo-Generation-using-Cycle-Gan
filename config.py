import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/content"
CHECKPOINT_SAVE_DIR = "/content/drive/MyDrive/Academics/GAN/saved_checkpoints"
SAVED_IMAGE_DIR = "/content/drive/MyDrive/Academics/GAN/saved_image"
VAL_DIR = "data/val"
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.2
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_P = "gen_p.pth.tar"
CHECKPOINT_GEN_M = "gen_m.pth.tar"
CHECKPOINT_CRITIC_P = "critic_p.pth.tar"
CHECKPOINT_CRITIC_M = "critic_m.pth.tar"
IMAGE_SIZE = 256

transforms = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
