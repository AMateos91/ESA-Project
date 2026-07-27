from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from models.unet import FireUNet
from config import (
    MODEL_DIR,
    PROCESSED_DIR,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("Evaluation")



class EvaluationError(Exception):
    pass



class FireEvaluator:


    def __init__(
        self,
        model_path: Path,
        device=None,
    ):

        self.model_path = Path(
            model_path
        )


        self.device = (
            device
            if device
            else
            torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )


        self.model = FireUNet(
            in_channels=11,
            out_channels=1
        )


        self.load_model()



    def load_model(self):

        if not self.model_path.exists():

            raise EvaluationError(
                f"Model not found: {self.model_path}"
            )


        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )


        self.model.load_state_dict(
            checkpoint["model_state"]
        )


        self.model.to(
            self.device
        )


        self.model.eval()


        logger.info(
            "Model loaded successfully"
        )



    def predict(
        self,
        image: np.ndarray
    ):


        if image.ndim != 3:

            raise EvaluationError(
                "Input must have shape (bands,height,width)"
            )


        tensor = torch.from_numpy(
            image
        ).float()


        tensor = tensor.unsqueeze(
            0
        )


        tensor = tensor.to(
            self.device
        )


        with torch.no_grad():

            prediction = self.model(
                tensor
            )


        prediction = (
            prediction
            .squeeze()
            .cpu()
            .numpy()
        )


        return prediction



    def threshold(
        self,
        prediction,
        value=0.5
    ):

        return (
            prediction >= value
        ).astype(
            np.uint8
        )



    def save_prediction(
        self,
        prediction,
        output_path: Path
    ):


        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        np.save(
            output_path,
            prediction
        )


        logger.info(
            f"Prediction saved: {output_path}"
        )



def load_processed_image(
    path: Path
):

    image = np.load(
        path
    )

    return image



def main():

    model_path = (
        MODEL_DIR /
        "best_fire_unet.pth"
    )


    image_path = (
        PROCESSED_DIR /
        "sample_00000_image.npy"
    )


    output_path = (
        PROCESSED_DIR /
        "prediction_00000.npy"
    )


    image = load_processed_image(
        image_path
    )


    evaluator = FireEvaluator(
        model_path
    )


    probability = evaluator.predict(
        image
    )


    evaluator.save_prediction(
        probability,
        output_path
    )


    mask = evaluator.threshold(
        probability
    )


    evaluator.save_prediction(
        mask,
        PROCESSED_DIR /
        "fire_mask_prediction.npy"
    )



if __name__ == "__main__":

    main()
