from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rasterio.transform import from_origin
from rasterio.features import rasterize


from config import (
    PROCESSED_DIR,
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("Fusion")



class FusionError(Exception):
    pass



class WeatherFusion:


    def __init__(
        self,
        output_dir: Path = PROCESSED_DIR
    ):

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )



    def load_weather(
        self,
        csv_path: Path
    ):


        if not csv_path.exists():

            raise FusionError(
                f"Missing file {csv_path}"
            )


        return pd.read_csv(
            csv_path
        )



    def create_grid(
        self,
        height: int,
        width: int,
        bounds
    ):


        west, south, east, north = bounds


        pixel_width = (
            east - west
        ) / width


        pixel_height = (
            north - south
        ) / height


        transform = from_origin(
            west,
            north,
            pixel_width,
            pixel_height
        )


        return transform



    def rasterize_variable(
        self,
        dataframe,
        variable,
        height,
        width,
        bounds
    ):


        transform = self.create_grid(
            height,
            width,
            bounds
        )


        shapes = []


        for _, row in dataframe.iterrows():

            if (
                pd.isna(row[variable])
                or
                pd.isna(row["lat"])
                or
                pd.isna(row["lon"])
            ):
                continue


            point = (
                row["lon"],
                row["lat"]
            )


            value = float(
                row[variable]
            )


            shapes.append(
                (
                    {
                        "type": "Point",
                        "coordinates": point
                    },
                    value
                )
            )


        raster = rasterize(
            shapes,
            out_shape=(
                height,
                width
            ),
            transform=transform,
            fill=0,
            dtype=np.float32
        )


        return raster



    def create_weather_stack(
        self,
        dataframe,
        image_shape,
        bounds
    ):


        height = image_shape[0]

        width = image_shape[1]


        variables = [
            "ta",
            "hr",
            "vv",
            "prec"
        ]


        layers = []


        for variable in variables:


            if variable not in dataframe.columns:

                layer = np.zeros(
                    (
                        height,
                        width
                    ),
                    dtype=np.float32
                )

            else:

                layer = self.rasterize_variable(
                    dataframe,
                    variable,
                    height,
                    width,
                    bounds
                )


            layers.append(
                layer
            )


        return np.stack(
            layers,
            axis=0
        )



    def merge(
        self,
        modis_image,
        weather_stack
    ):


        if (
            modis_image.shape[1:]
            !=
            weather_stack.shape[1:]
        ):

            raise FusionError(
                "Spatial dimensions do not match"
            )


        fused = np.concatenate(
            [
                modis_image,
                weather_stack
            ],
            axis=0
        )


        return fused



    def save(
        self,
        array,
        name
    ):


        path = (
            self.output_dir /
            name
        )


        np.save(
            path,
            array
        )


        logger.info(
            f"Saved {path}"
        )



def main():


    processor = WeatherFusion()


    modis_file = (
        PROCESSED_DIR /
        "sample_00000_image.npy"
    )


    weather_file = (
        PROCESSED_DIR /
        "aemet_weather.csv"
    )


    image = np.load(
        modis_file
    )


    weather = processor.load_weather(
        weather_file
    )


    weather_stack = processor.create_weather_stack(
        weather,
        image.shape[1:],
        (
            -9.5,
            35.8,
            4.5,
            43.8
        )
    )


    fused = processor.merge(
        image,
        weather_stack
    )


    processor.save(
        fused,
        "sample_00000_fused.npy"
    )



if __name__ == "__main__":

    main()
