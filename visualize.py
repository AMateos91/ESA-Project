from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


from config import PROCESSED_DIR



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("Visualization")



class VisualizationError(Exception):
    pass



def load_array(
    path: Path
):

    if not path.exists():

        raise VisualizationError(
            f"File not found: {path}"
        )

    return np.load(
        path
    )



def normalize_band(
    band
):

    minimum = np.min(
        band
    )

    maximum = np.max(
        band
    )

    if maximum == minimum:

        return np.zeros_like(
            band
        )

    return (
        band - minimum
    ) / (
        maximum - minimum
    )



def create_rgb(
    image
):

    if image.shape[0] < 3:

        raise VisualizationError(
            "Image needs at least 3 bands"
        )


    rgb = np.stack(
        [
            normalize_band(image[0]),
            normalize_band(image[1]),
            normalize_band(image[2]),
        ],
        axis=-1
    )


    return rgb



def plot_prediction(
    image,
    prediction,
    output_path: Path,
):


    rgb = create_rgb(
        image
    )


    plt.figure(
        figsize=(10,10)
    )


    plt.imshow(
        rgb
    )


    plt.imshow(
        prediction,
        cmap="hot",
        alpha=0.45,
        vmin=0,
        vmax=1
    )


    plt.colorbar(
        label="Fire probability"
    )


    plt.axis(
        "off"
    )


    plt.tight_layout()


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    logger.info(
        f"Saved: {output_path}"
    )



def main():


    image_path = (
        PROCESSED_DIR /
        "sample_00000_image.npy"
    )


    prediction_path = (
        PROCESSED_DIR /
        "prediction_00000.npy"
    )


    output_path = (
        PROCESSED_DIR /
        "fire_prediction.png"
    )


    image = load_array(
        image_path
    )


    prediction = load_array(
        prediction_path
    )


    plot_prediction(
        image,
        prediction,
        output_path
    )



if __name__ == "__main__":

    main()
