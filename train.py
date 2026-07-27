from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import (
    MODISFireDataset,
    split_dataset,
)

from models.unet import FireUNet

from config import (
    TRAINING,
    MODEL_DIR,
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("Training")



class TrainerError(Exception):
    pass



class FireTrainer:


    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
    ):

        self.model = model.to(
            device
        )

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.device = device


        self.loss_function = nn.BCELoss()


        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=TRAINING.learning_rate
        )


        self.best_loss = float(
            "inf"
        )



    def train_epoch(self):

        self.model.train()

        total_loss = 0.0


        for images, masks in self.train_loader:


            images = images.to(
                self.device
            )

            masks = masks.to(
                self.device
            )


            predictions = self.model(
                images
            )


            loss = self.loss_function(
                predictions,
                masks
            )


            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()


            total_loss += (
                loss.item()
                *
                images.size(0)
            )


        return (
            total_loss /
            len(self.train_loader.dataset)
        )



    def validate(self):

        self.model.eval()

        total_loss = 0.0


        with torch.no_grad():


            for images, masks in self.val_loader:


                images = images.to(
                    self.device
                )

                masks = masks.to(
                    self.device
                )


                predictions = self.model(
                    images
                )


                loss = self.loss_function(
                    predictions,
                    masks
                )


                total_loss += (
                    loss.item()
                    *
                    images.size(0)
                )


        return (
            total_loss /
            len(self.val_loader.dataset)
        )



    def save_model(
        self,
        loss,
        epoch
    ):


        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        filename = (
            MODEL_DIR /
            "best_fire_unet.pth"
        )


        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "model_state": self.model.state_dict(),
            },
            filename
        )


        logger.info(
            f"Model saved: {filename}"
        )



    def fit(
        self,
        epochs
    ):


        for epoch in range(
            epochs
        ):


            train_loss = (
                self.train_epoch()
            )


            val_loss = (
                self.validate()
            )


            logger.info(
                f"Epoch {epoch+1}/{epochs} "
                f"Train={train_loss:.5f} "
                f"Val={val_loss:.5f}"
            )


            if val_loss < self.best_loss:

                self.best_loss = val_loss

                self.save_model(
                    val_loss,
                    epoch
                )



def create_loaders():


    dataset = MODISFireDataset()


    train_set, val_set = split_dataset(
        dataset
    )


    train_loader = DataLoader(
        train_set,
        batch_size=TRAINING.batch_size,
        shuffle=True,
        num_workers=TRAINING.workers,
        pin_memory=True,
    )


    val_loader = DataLoader(
        val_set,
        batch_size=TRAINING.batch_size,
        shuffle=False,
        num_workers=TRAINING.workers,
        pin_memory=True,
    )


    return (
        train_loader,
        val_loader
    )



def main():


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )


    logger.info(
        f"Using device: {device}"
    )


    train_loader, val_loader = (
        create_loaders()
    )


    model = FireUNet(
        in_channels=11,
        out_channels=1
    )


    trainer = FireTrainer(
        model,
        train_loader,
        val_loader,
        device
    )


    trainer.fit(
        epochs=TRAINING.epochs
    )



if __name__ == "__main__":

    main()
