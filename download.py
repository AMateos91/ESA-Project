from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from earthaccess import Auth, DataCollections, DataGranules, Store

from config import (
    RAW_DIR,
    MODIS_IMAGE_PRODUCT,
    MODIS_FIRE_PRODUCT,
    MODIS_VERSION,
    START_DATE,
    END_DATE,
    BOUNDING_BOX,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("MODISDownloader")


class MODISDownloadError(Exception):
    pass


class CollectionNotFoundError(MODISDownloadError):
    pass


class AuthenticationError(MODISDownloadError):
    pass


class MODISDownloader:

    def __init__(
        self,
        output_dir: Path = RAW_DIR,
        version: str = MODIS_VERSION,
    ):

        self.output_dir = Path(output_dir)
        self.version = version
        self.auth: Optional[Auth] = None

        self.image_dir = (
            self.output_dir /
            MODIS_IMAGE_PRODUCT
        )

        self.fire_dir = (
            self.output_dir /
            MODIS_FIRE_PRODUCT
        )

        self.image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.fire_dir.mkdir(
            parents=True,
            exist_ok=True
        )


    def authenticate(self) -> None:

        try:

            self.auth = Auth().login(
                strategy="interactive",
                persist=True
            )

        except Exception as exc:

            raise AuthenticationError(
                str(exc)
            )

        if not self.auth.authenticated:

            raise AuthenticationError(
                "Earthdata authentication failed"
            )

        logger.info(
            "Earthdata authentication successful"
        )


    def get_collection(
        self,
        short_name: str
    ):

        if self.auth is None:

            raise AuthenticationError(
                "Authenticate first"
            )

        collections = (
            DataCollections(self.auth)
            .short_name(short_name)
            .version(self.version)
            .get()
        )

        if not collections:

            raise CollectionNotFoundError(
                f"{short_name} not found"
            )

        return collections[0]


    def get_concept_id(
        self,
        short_name: str
    ) -> str:

        collection = self.get_collection(
            short_name
        )

        return collection["meta"]["concept-id"]


    def search_granules(
        self,
        product: str,
        start_date: str,
        end_date: str,
        bbox: tuple[
            float,
            float,
            float,
            float
        ],
    ):

        if self.auth is None:

            raise AuthenticationError(
                "Authenticate first"
            )

        concept_id = self.get_concept_id(
            product
        )

        query = (
            DataGranules(self.auth)
            .concept_id(concept_id)
            .temporal(
                start_date,
                end_date
            )
            .bounding_box(
                *bbox
            )
        )

        if query.hits() == 0:

            return []

        return query.get()


    def download_granules(
        self,
        granules,
        destination: Path,
    ):

        if not granules:

            return

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        store = Store(self.auth)

        store.get(
            granules,
            local_path=str(destination)
        )


    def download_with_retry(
        self,
        granules,
        destination: Path,
        retries: int = 3,
    ):

        error = None

        for attempt in range(retries):

            try:

                self.download_granules(
                    granules,
                    destination
                )

                return

            except Exception as exc:

                error = exc

                logger.warning(
                    f"Download attempt {attempt + 1} failed: {exc}"
                )

        raise MODISDownloadError(
            f"Download failed: {error}"
        )


    def download_product(
        self,
        product: str,
        start_date: str,
        end_date: str,
        bbox: tuple[
            float,
            float,
            float,
            float
        ],
        destination: Path,
    ):

        granules = self.search_granules(
            product,
            start_date,
            end_date,
            bbox
        )

        if not granules:

            logger.warning(
                f"No granules found for {product}"
            )

            return []

        self.download_with_retry(
            granules,
            destination
        )

        return granules


    def download_dataset(
        self,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        bbox: tuple[
            float,
            float,
            float,
            float
        ] = BOUNDING_BOX,
    ):

        if self.auth is None:

            self.authenticate()


        images = self.download_product(
            MODIS_IMAGE_PRODUCT,
            start_date,
            end_date,
            bbox,
            self.image_dir
        )


        fire_masks = self.download_product(
            MODIS_FIRE_PRODUCT,
            start_date,
            end_date,
            bbox,
            self.fire_dir
        )


        return {
            "images": images,
            "fire_masks": fire_masks,
        }


    def existing_files(
        self,
        directory: Path
    ) -> list[Path]:

        if not directory.exists():

            return []

        return [
            file
            for file in directory.rglob("*")
            if file.is_file()
        ]


    def validate_dataset(self) -> bool:

        images = self.existing_files(
            self.image_dir
        )

        masks = self.existing_files(
            self.fire_dir
        )

        return (
            len(images) > 0
            and
            len(masks) > 0
        )


    def summary(self):

        images = self.existing_files(
            self.image_dir
        )

        masks = self.existing_files(
            self.fire_dir
        )

        logger.info(
            f"MOD09GA files: {len(images)}"
        )

        logger.info(
            f"MOD14A1 files: {len(masks)}"
        )

        logger.info(
            f"Output: {self.output_dir}"
        )


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default=START_DATE
    )

    parser.add_argument(
        "--end",
        default=END_DATE
    )

    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=BOUNDING_BOX
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    downloader = MODISDownloader()

    downloader.download_dataset(
        start_date=args.start,
        end_date=args.end,
        bbox=tuple(args.bbox),
    )

    if downloader.validate_dataset():

        downloader.summary()

    else:

        raise SystemExit(
            "Dataset validation failed"
        )


if __name__ == "__main__":

    main()
