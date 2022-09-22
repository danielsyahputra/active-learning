from utils.download import download_data
from utils.torch import dataloader

def main() -> None:
    dataloaders = dataloader(train_batch_size=64, test_batch_size=16)
    print(dataloaders)

if __name__=="__main__":
    main()