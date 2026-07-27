from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset, random_split


from config import (
    PROCESSED_DIR,
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("Dataset")



class DatasetError(Exception):
    pass



class MODISFireDataset(Dataset):


    def __init__(
        self,
        root_dir: Path = PROCESSED_DIR,
        normalize: bool = True,
    ):


        self.root_dir = Path(
            root_dir
        )


        self.normalize = normalize


        self.images = sorted(
            self.root_dir.glob(
                "*_fused.npy"
            )
        )


        self.masks = sorted(
            self.root_dir.glob(
                "*_mask.npy"
            )
        )


        if len(self.images) == 0:

            raise DatasetError(
                "No fused images found"
            )


        if len(self.images) != len(self.masks):

            raise DatasetError(
                "Images and masks count mismatch"
            )


        logger.info(
            f"Dataset loaded: {len(self.images)} samples"
        )



    def __len__(self):

        return len(
            self.images
        )



    def normalize_image(
        self,
        image
    ):


        channels = []


        for band in image:


            minimum = np.min(
                band
            )


            maximum = np.max(
                band
            )


            if maximum == minimum:

                channels.append(
                    np.zeros_like(
                        band
                    )
                )

            else:

                channels.append(
                    (
                        band - minimum
                    )
                    /
                    (
                        maximum - minimum
                    )
                )


        return np.stack(
            channels,
            axis=0
        )



    def __getitem__(
        self,
        index
    ):


        image_path = (
            self.images[index]
        )


        mask_path = (
            self.masks[index]
        )


        image = np.load(
            image_path
        ).astype(
            np.float32
        )


        mask = np.load(
            mask_path
        ).astype(
            np.float32
        )



        if self.normalize:

            image = self.normalize_image(
                image
            )



        image_tensor = torch.from_numpy(
            image
        )


        mask_tensor = torch.from_numpy(
            mask
        )


        if mask_tensor.ndim == 2:

            mask_tensor = (
                mask_tensor
                .unsqueeze(0)
            )



        return (
            image_tensor,
            mask_tensor
        )



def split_dataset(
    dataset,
    train_fraction=0.8
):


    train_size = int(
        len(dataset)
        *
        train_fraction
    )


    val_size = (
        len(dataset)
        -
        train_size
    )


    return random_split(
        dataset,
        [
            train_size,
            val_size
        ]
    )



if __name__ == "__main__":


    dataset = MODISFireDataset()


    image, mask = dataset[0]


    print(
        "Image:",
        image.shape
    )


    print(
        "Mask:",
        mask.shape
    )
