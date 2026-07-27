from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


from config import (
    PROCESSED_DIR,
    TRAINING,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("MODISDataset")



class DatasetError(Exception):
    pass



class MODISFireDataset(Dataset):


    def __init__(
        self,
        data_dir: Path = PROCESSED_DIR,
        transform=None,
    ):

        self.data_dir = Path(
            data_dir
        )

        self.transform = transform


        self.images = sorted(
            self.data_dir.glob(
                "*_image.npy"
            )
        )


        self.masks = sorted(
            self.data_dir.glob(
                "*_mask.npy"
            )
        )


        if len(self.images) == 0:

            raise DatasetError(
                "No image files found"
            )


        if len(self.images) != len(self.masks):

            raise DatasetError(
                "Images and masks mismatch"
            )


        logger.info(
            f"Dataset size: {len(self.images)}"
        )



    def __len__(self):

        return len(
            self.images
        )



    def __getitem__(
        self,
        index
    ):


        image = np.load(
            self.images[index]
        )


        mask = np.load(
            self.masks[index]
        )


        image = torch.from_numpy(
            image
        ).float()


        mask = torch.from_numpy(
            mask
        ).float()


        if mask.ndim == 2:

            mask = mask.unsqueeze(
                0
            )


        if self.transform:

            image = self.transform(
                image
            )


        return (
            image,
            mask
        )



def create_dataloader(
    batch_size: int = TRAINING.batch_size,
    shuffle: bool = True,
    workers: int = 4,
):


    dataset = MODISFireDataset()


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )


    return loader



def split_dataset(
    dataset,
    train_fraction=0.8
):

    total = len(
        dataset
    )

    train_size = int(
        total *
        train_fraction
    )


    val_size = (
        total -
        train_size
    )


    return torch.utils.data.random_split(
        dataset,
        [
            train_size,
            val_size
        ]
    )



if __name__ == "__main__":


    dataset = MODISFireDataset()


    train_set, val_set = split_dataset(
        dataset
    )


    logger.info(
        f"Train samples: {len(train_set)}"
    )


    logger.info(
        f"Validation samples: {len(val_set)}"
    )


    loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True
    )


    images, masks = next(
        iter(loader)
    )


    print(
        images.shape
    )

    print(
        masks.shape
    )
