from typing import Tuple
from sklearn.model_selection import train_test_split
from utils import *
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset, DataLoader

class CustomSubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

def get_dataloader(train_batch_size: int = 64, test_batch_size: int = 16) -> Tuple:
    data_dirs = get_data_dirs()
    amazon, dslr, webcam = get_dataset(data_dirs=data_dirs, transform=get_transform())
    domains = [amazon, dslr, webcam]
    names = ['amazon', 'dslr', 'webcam']

    train = {}
    val = {}
    for name, domain in zip(names, domains):
        train_indices, val_indices = train_test_split(list(range(len(domain.targets))), 
                                                test_size=0.3,
                                                stratify=domain.targets,
                                                random_state=42)
        train_targets = [domain.targets[idx] for idx in train_indices]
        val_targets = [domain.targets[idx] for idx in val_indices]
        train_domain = CustomSubset(domain, train_indices, train_targets)
        val_domain = CustomSubset(domain, val_indices, val_targets)

        train[name] = train_domain
        val[name] = val_domain

    # Get Training Data
    train_data = ConcatDataset([train[k] for k in train.keys()])

    # Split Val to Val - Test
    val_data = ConcatDataset([val[k] for k in val.keys()])
    targets = val_data.datasets[0].targets.copy()
    targets.extend(val_data.datasets[1].targets.copy())
    targets.extend(val_data.datasets[2].targets.copy())
    val_indices, test_indices = train_test_split(list(range(len(targets))),
                                            test_size=0.3, stratify=targets, random_state=42)
    val_targets = [targets[idx] for idx in val_indices]
    test_targets = [targets[idx] for idx in test_indices]

    # Get Val and Test
    fix_val_data = CustomSubset(val_data, val_indices, val_targets)
    test_data = CustomSubset(val_data, test_indices, test_targets)

    num_workers = 2
    train_dataloader = DataLoader(dataset=train_data,
                                    batch_size=train_batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)
    val_dataloader = DataLoader(dataset=fix_val_data,
                                    batch_size=test_batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_data,
                                    batch_size=test_batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)
    datas = {"train": train_data, "val": fix_val_data, "test": test_data}
    loaders = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}
    return datas, loaders
