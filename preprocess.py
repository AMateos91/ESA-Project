from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import geopandas as gpd


from config import (
    RAW_DIR,
    PROCESSED_DIR,
    TRAINING,
    MODIS_IMAGE_PRODUCT,
    MODIS_FIRE_PRODUCT,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("Preprocess")


class PreprocessError(Exception):
    pass



class MODISPreprocessor:


    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        output_dir: Path = PROCESSED_DIR,
    ):

        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

        self.image_dir = (
            self.raw_dir /
            MODIS_IMAGE_PRODUCT
        )

        self.fire_dir = (
            self.raw_dir /
            MODIS_FIRE_PRODUCT
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


    def list_files(
        self,
        directory: Path,
        extension=".hdf"
    ):

        if not directory.exists():

            return []

        return sorted(
            directory.rglob(
                f"*{extension}"
            )
        )



    def read_hdf(
        self,
        file_path: Path
    ) -> np.ndarray:

        try:

            with rasterio.open(
                file_path
            ) as src:

                if not src.subdatasets:

                    data = src.read()

                else:

                    with rasterio.open(
                        src.subdatasets[0]
                    ) as sub:

                        data = sub.read()


            return data.astype(
                np.float32
            )


        except Exception as exc:

            raise PreprocessError(
                f"Cannot read {file_path}: {exc}"
            )



    def normalize(
        self,
        image: np.ndarray
    ) -> np.ndarray:


        minimum = np.nanmin(
            image
        )

        maximum = np.nanmax(
            image
        )


        if maximum == minimum:

            return np.zeros_like(
                image
            )


        image = (
            image - minimum
        ) / (
            maximum - minimum
        )


        image = np.nan_to_num(
            image,
            nan=0.0
        )


        return image



    def create_empty_mask(
        self,
        shape: Tuple[int,int]
    ) -> np.ndarray:


        return np.zeros(
            shape,
            dtype=np.uint8
        )



    def read_fire_mask(
        self,
        file_path: Path,
        shape: Tuple[int,int]
    ) -> np.ndarray:


        fire = self.read_hdf(
            file_path
        )


        fire = np.squeeze(
            fire
        )


        if fire.shape != shape:

            fire = np.resize(
                fire,
                shape
            )


        mask = np.zeros_like(
            fire,
            dtype=np.uint8
        )


        mask[
            fire > 7
        ] = 1


        return mask



    def crop_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        size: int = TRAINING.image_size
    ):

        bands, height, width = image.shape


        if height < size or width < size:

            padded_image = np.zeros(
                (
                    bands,
                    max(height,size),
                    max(width,size)
                ),
                dtype=image.dtype
            )


            padded_mask = np.zeros(
                (
                    max(height,size),
                    max(width,size)
                ),
                dtype=mask.dtype
            )


            padded_image[
                :,
                :height,
                :width
            ] = image


            padded_mask[
                :height,
                :width
            ] = mask


            image = padded_image
            mask = padded_mask


            height = image.shape[1]
            width = image.shape[2]


        y = (
            height - size
        ) // 2

        x = (
            width - size
        ) // 2


        image_patch = (
            image[
                :,
                y:y+size,
                x:x+size
            ]
        )


        mask_patch = (
            mask[
                y:y+size,
                x:x+size
            ]
        )


        return (
            image_patch,
            mask_patch
        )



    def process_pair(
        self,
        image_file: Path,
        fire_file: Path,
    ):

        image = self.read_hdf(
            image_file
        )


        image = self.normalize(
            image
        )


        mask = self.read_fire_mask(
            fire_file,
            (
                image.shape[1],
                image.shape[2]
            )
        )


        image, mask = self.crop_patch(
            image,
            mask
        )


        return image, mask



    def save_pair(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        name: str
    ):


        np.save(
            self.output_dir /
            f"{name}_image.npy",
            image
        )


        np.save(
            self.output_dir /
            f"{name}_mask.npy",
            mask
        )



    def run(self):


        images = self.list_files(
            self.image_dir
        )


        fires = self.list_files(
            self.fire_dir
        )


        if not images:

            raise PreprocessError(
                "No MOD09GA files found"
            )


        if not fires:

            raise PreprocessError(
                "No MOD14A1 files found"
            )


        count = min(
            len(images),
            len(fires)
        )


        for index in range(count):

            image, mask = self.process_pair(
                images[index],
                fires[index]
            )


            self.save_pair(
                image,
                mask,
                f"sample_{index:05d}"
            )


            logger.info(
                f"Processed sample {index}"
            )



def main():

    processor = MODISPreprocessor()

    processor.run()


if __name__ == "__main__":

    main()
