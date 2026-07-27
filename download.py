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
        
    # ========================================================
    # BÚSQUEDA DE GRANOS
    # ========================================================

    def search_granules(
        self,
        short_name: str,
        start_date: str,
        end_date: str,
        bbox: tuple[float, float, float, float],
    ):
        """
        Busca archivos MODIS disponibles.

        Parameters
        ----------
        short_name:
            Producto MODIS.

        start_date:
            Fecha inicial YYYY-MM-DD.

        end_date:
            Fecha final YYYY-MM-DD.

        bbox:
            (west, south, east, north)

        Returns
        -------
        lista de granulos
        """

        if self.auth is None:
            raise AuthenticationError(
                "Debe autenticarse antes"
            )


        concept_id = self.get_concept_id(
            short_name
        )


        logger.info(
            f"Buscando granos para {short_name}"
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


        hits = query.hits()


        logger.info(
            f"{short_name}: {hits} granos encontrados"
        )


        if hits == 0:
            logger.warning(
                f"No hay datos disponibles para {short_name}"
            )

            return []


        granules = query.get()


        return granules



    # ========================================================
    # INFORMACIÓN DE GRÁNULOS
    # ========================================================

    def print_granule_summary(
        self,
        granules,
        name: str
    ):
        """
        Imprime información básica de los archivos encontrados.
        """


        logger.info(
            f"Resumen {name}"
        )


        for idx, granule in enumerate(
            granules
        ):

            try:

                title = (
                    granule["umm"]
                    ["GranuleUR"]
                )

            except Exception:

                title = "desconocido"


            logger.info(
                f"{idx+1}: {title}"
            )



    # ========================================================
    # DESCARGA
    # ========================================================

    def download_granules(
        self,
        granules,
        destination: Path
    ):
        """
        Descarga una lista de granulos.

        Parameters
        ----------
        granules:
            Resultado de DataGranules.get()

        destination:
            Carpeta destino.
        """


        if not granules:

            logger.warning(
                "Lista de granos vacía"
            )

            return



        destination.mkdir(
            parents=True,
            exist_ok=True
        )


        logger.info(
            f"Descargando {len(granules)} archivos"
        )


        store = Store(
            self.auth
        )


        try:

            store.get(
                granules,
                local_path=str(destination)
            )


        except Exception as exc:

            raise MODISDownloadError(
                f"Error durante descarga: {exc}"
            )


        logger.info(
            "Descarga completada"
        )



    # ========================================================
    # DESCARGA DE PRODUCTO COMPLETO
    # ========================================================

    def download_product(
        self,
        product: str,
        start_date: str,
        end_date: str,
        bbox: tuple[float, float, float, float],
        destination: Path,
    ):
        """
        Flujo completo:

        1. Buscar granos
        2. Mostrar información
        3. Descargar
        """


        granules = self.search_granules(
            short_name=product,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox
        )


        if not granules:

            logger.warning(
                f"No se descargará {product}"
            )

            return []


        self.print_granule_summary(
            granules,
            product
        )


        self.download_granules(
            granules,
            destination
        )


        return granules



    # ========================================================
    # DESCARGA DEL DATASET COMPLETO
    # ========================================================

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
        """
        Descarga todos los productos necesarios
        para entrenar el detector.

        Descarga:

            MOD09GA -> imágenes

            MOD14A1 -> etiquetas
        """


        if self.auth is None:

            self.authenticate()



        logger.info(
            "Inicio descarga dataset MODIS"
        )


        image_granules = (
            self.download_product(
                product=MODIS_IMAGE_PRODUCT,
                start_date=start_date,
                end_date=end_date,
                bbox=bbox,
                destination=self.image_dir
            )
        )


        fire_granules = (
            self.download_product(
                product=MODIS_FIRE_PRODUCT,
                start_date=start_date,
                end_date=end_date,
                bbox=bbox,
                destination=self.fire_dir
            )
        )


        logger.info(
            "Proceso de descarga finalizado"
        )


        return {
            "images": image_granules,
            "fire_masks": fire_granules,
        }
