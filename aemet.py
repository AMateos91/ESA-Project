from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd


from config import (
    AEMET_API_KEY,
    PROCESSED_DIR,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("AEMET")



class AEMETError(Exception):
    pass



class AEMETClient:


    BASE_URL = (
        "https://opendata.aemet.es"
        "/opendata/api"
    )


    def __init__(
        self,
        api_key: str = AEMET_API_KEY
    ):

        self.api_key = api_key

        self.headers = {
            "api_key": self.api_key
        }



    def request_metadata(
        self,
        endpoint: str
    ):


        url = (
            self.BASE_URL +
            endpoint
        )


        response = requests.get(
            url,
            headers=self.headers
        )


        if response.status_code != 200:

            raise AEMETError(
                response.text
            )


        data = response.json()


        return data["datos"]



    def get_station_data(
        self,
        station_id: str
    ):


        endpoint = (
            "/observacion/convencional/"
            f"datos/estacion/{station_id}"
        )


        url = self.request_metadata(
            endpoint
        )


        response = requests.get(
            url
        )


        return response.json()



    def get_daily_data(
        self,
        station_id: str,
        date: str
    ):


        endpoint = (
            "/valores/climatologicos/"
            "diarios/datos/"
            f"fecha/{date}/"
            f"estacion/{station_id}"
        )


        url = self.request_metadata(
            endpoint
        )


        response = requests.get(
            url
        )


        return response.json()



    def dataframe(
        self,
        data
    ):


        df = pd.DataFrame(
            data
        )


        return df



    def save(
        self,
        dataframe,
        filename="aemet_weather.csv"
    ):


        output = (
            PROCESSED_DIR /
            filename
        )


        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        dataframe.to_csv(
            output,
            index=False
        )


        logger.info(
            f"Saved {output}"
        )



def main():


    client = AEMETClient()


    station = "3195"


    data = client.get_station_data(
        station
    )


    df = client.dataframe(
        data
    )


    client.save(
        df
    )


    print(
        df.head()
    )



if __name__ == "__main__":

    main()
