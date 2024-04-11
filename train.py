import torch
from dataset import PhotoMonetDataset
from discriminator import Discriminator
from generator import Generator
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image


def train_fn(): ...


def main(): ...


if __name__ == "__main__":
    main()
