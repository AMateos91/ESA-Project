"""
download.py

Módulo de descarga de productos MODIS mediante Earthaccess.

Productos utilizados:
    - MOD09GA : reflectancia superficial MODIS (entrada del modelo)
    - MOD14A1 : máscara de incendios MODIS (etiquetas)

Requiere:
    pip install earthaccess

"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


import earthaccess
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


# ============================================================
# LOGGING
# ============================================================

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


logger = logging.getLogger("MODISDownloader")



# ============================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================

class MODISDownloadError(Exception):
    """
    Error general durante la descarga MODIS.
    """
    pass



class CollectionNotFoundError(MODISDownloadError):
    """
    No se encontró una colección MODIS.
    """
    pass



class AuthenticationError(MODISDownloadError):
    """
    Fallo de autenticación Earthdata.
    """
    pass



# ============================================================
# CLASE PRINCIPAL
# ============================================================

class MODISDownloader:
    """
    Cliente encargado de descargar productos MODIS.

    Ejemplo:

        downloader = MODISDownloader()

        downloader.download(
            start_date="2020-08-01",
            end_date="2020-08-10",
            bbox=(10,45,15,48)
        )

    """

    def __init__(
        self,
        output_dir: Path = RAW_DIR,
        version: str = MODIS_VERSION,
    ):

        self.output_dir = Path(output_dir)

        self.version = version

        self.auth: Optional[Auth] = None


        # Crear estructura

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


        logger.info(
            "MODISDownloader inicializado"
        )



    # ========================================================
    # AUTENTICACIÓN
    # ========================================================

    def authenticate(self) -> None:
        """
        Autentica contra NASA Earthdata.

        Usa el sistema interactivo de Earthaccess.
        """

        logger.info(
            "Autenticando con NASA Earthdata..."
        )


        try:

            self.auth = Auth().login(
                strategy="interactive",
                persist=True
            )


        except Exception as exc:

            raise AuthenticationError(
                f"No fue posible autenticar: {exc}"
            )


        if not self.auth.authenticated:

            raise AuthenticationError(
                "Earthdata rechazó la autenticación"
            )


        logger.info(
            "Autenticación correcta"
        )



    # ========================================================
    # COLECCIONES
    # ========================================================

    def get_collection(
        self,
        short_name: str
    ):
        """
        Obtiene una colección MODIS concreta.

        Parameters
        ----------
        short_name:
            Nombre corto del producto.

        Returns
        -------
        Collection Earthaccess
        """


        if self.auth is None:

            raise AuthenticationError(
                "Debe autenticarse antes"
            )


        logger.info(
            f"Buscando colección {short_name}"
        )


        collections = (
            DataCollections(self.auth)
            .short_name(short_name)
            .version(self.version)
            .get()
        )


        if not collections:

            raise CollectionNotFoundError(
                f"No existe {short_name} versión {self.version}"
            )


        collection = collections[0]


        logger.info(
            "Colección encontrada:"
            f" {collection['umm']['EntryTitle']}"
        )


        return collection



    # ========================================================
    # OBTENER CONCEPT ID
    # ========================================================

    def get_concept_id(
        self,
        short_name: str
    ) -> str:
        """
        Obtiene el concept-id necesario
        para consultar granos.
        """


        collection = self.get_collection(
            short_name
        )


        concept_id = (
            collection["meta"]["concept-id"]
        )


        logger.info(
            f"{short_name} concept-id: {concept_id}"
        )


        return concept_id

