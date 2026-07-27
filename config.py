"""
config.py
Configuración global del proyecto Fire Detection.
"""

from pathlib import Path
from dataclasses import dataclass


# ==========================
# DIRECTORIOS
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# PRODUCTOS MODIS
# ==========================

MODIS_IMAGE_PRODUCT = "MOD09GA"

MODIS_FIRE_PRODUCT = "MOD14A1"

MODIS_VERSION = "061"


# ==========================
# FECHAS
# ==========================

START_DATE = "2026-07-01"
END_DATE = "2026-07-27"


# ==========================
# BOUNDING BOX
# ==========================
# xmin, ymin, xmax, ymax

BOUNDING_BOX = (
    10.0,
    45.0,
    15.0,
    48.0,
)


# ==========================
# ENTRENAMIENTO
# ==========================

@dataclass
class TrainingConfig:

    image_size: int = 256

    batch_size: int = 8

    epochs: int = 50

    learning_rate: float = 1e-4

    num_workers: int = 4

    validation_split: float = 0.20

    random_seed: int = 42

    in_channels: int = 7
    out_channels: int = 1


TRAINING = TrainingConfig()


# ==========================
# HARDWARE
# ==========================

USE_GPU = True


# ==========================
# FORMATO
# ==========================

IMAGE_EXTENSION = ".hdf"

MODEL_NAME = "fire_unet.pt"
