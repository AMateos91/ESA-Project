from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio

from rasterio.features import rasterize
from rasterio.transform import from_origin

import geopandas as gpd


from config import (
    RAW_DIR,
    PROCESSED_DIR,
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

        self.raw_dir = Path(
            raw_dir
        )

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )



    def find_files(
        self,
        directory: Path,
        extension=".hdf"
    ):


        files = sorted(
            directory.rglob(
                f"*{extension}"
            )
        )


        if not files:

            raise PreprocessError(
                f"No files found in {directory}"
            )


        return files



    def read_raster(
        self,
        path: Path
    ):


        with rasterio.open(
            path
        ) as src:

            array = src.read()

            profile = src.profile


        return (
            array,
            profile
        )



    def normalize(
        self,
        image
    ):


        output = np.zeros_like(
            image,
            dtype=np.float32
        )


        for i in range(
            image.shape[0]
        ):


            band = image[i]


            minimum = np.nanmin(
                band
            )


            maximum = np.nanmax(
                band
            )


            if maximum == minimum:

                output[i] = 0

            else:

                output[i] = (
                    band - minimum
                ) / (
                    maximum - minimum
                )


        return output



    def create_mask(
        self,
        fire_file: Path,
        shape,
        transform=None
    ):


        with rasterio.open(
            fire_file
        ) as src:

            fire = src.read(
                1
            )

            profile = src.profile



        mask = np.where(
            fire > 0,
            1,
            0
        ).astype(
            np.float32
        )


        if mask.shape != shape:

            raise PreprocessError(
                "Image and fire mask dimensions differ"
            )


        return mask



    def save_pair(
        self,
        image,
        mask,
        index
    ):


        image_path = (
            self.output_dir /
            f"sample_{index:05d}_image.npy"
        )


        mask_path = (
            self.output_dir /
            f"sample_{index:05d}_mask.npy"
        )


        np.save(
            image_path,
            image
        )


        np.save(
            mask_path,
            mask
        )


        logger.info(
            f"Saved sample {index}"
        )



    def process(
        self
    ):


        image_dir = (
            self.raw_dir /
            MODIS_IMAGE_PRODUCT
        )


        fire_dir = (
            self.raw_dir /
            MODIS_FIRE_PRODUCT
        )


        images = self.find_files(
            image_dir
        )


        fires = self.find_files(
            fire_dir
        )


        if len(images) != len(fires):

            logger.warning(
                "Different number of images and masks"
            )


        total = min(
            len(images),
            len(fires)
        )


        for i in range(
            total
        ):


            logger.info(
                f"Processing {i+1}/{total}"
            )


            image, profile = self.read_raster(
                images[i]
            )


            image = image.astype(
                np.float32
            )


            image = self.normalize(
                image
            )


            mask = self.create_mask(
                fires[i],
                image.shape[1:]
            )


            self.save_pair(
                image,
                mask,
                i
            )


        logger.info(
            "Preprocessing completed"
        )



def main():


    processor = MODISPreprocessor()


    processor.process()



if __name__ == "__main__":

    main()
