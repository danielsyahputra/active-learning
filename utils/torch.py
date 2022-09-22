from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from pathlib import Path
from typing import Tuple

def get_transform() -> transforms.Compose:
    MEAN = (0.7932, 0.7864, 0.7827)
    STD = (0.3132, 0.3160, 0.3187)

    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    return transform

def get_data_dirs() -> Tuple:
    data_path = Path("data/")
    amazon_dir = data_path / "domain1/images"
    dslr_dir = data_path / "domain2/images"
    webcam_dir = data_path / "domain3/images"
    return amazon_dir, dslr_dir, webcam_dir

def get_dataset(data_dirs: Tuple, 
                transform: transforms.Compose) -> Tuple:
    amazon_dir, dslr_dir, webcam_dir = data_dirs
    amazon_dataset = datasets.ImageFolder(root=amazon_dir, transform=transform)
    dslr_dataset = datasets.ImageFolder(root=dslr_dir, transform=transform)
    webcam_dataset = datasets.ImageFolder(root=webcam_dir, transform=transform)
    return amazon_dataset, dslr_dataset, webcam_dataset

def dataloader(train_batch_size: int = 64,
                test_batch_size: int = 16,
                shuffle: bool = True,
                num_workers: int = 2) -> Tuple:
    data_dirs = get_data_dirs()
    amazon_dataset, dslr_dataset, webcam_dataset = get_dataset(data_dirs=data_dirs, transform=get_transform())
    amazon_dataloader = DataLoader(
                            dataset=amazon_dataset,
                            batch_size=train_batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers
                        )
    dslr_dataloader = DataLoader(dataset=dslr_dataset, batch_size=test_batch_size, shuffle=shuffle, num_workers=num_workers)
    webcam_dataloader = DataLoader(dataset=webcam_dataset, batch_size=test_batch_size, shuffle=shuffle, num_workers=num_workers)
    return amazon_dataloader, dslr_dataloader, webcam_dataloader