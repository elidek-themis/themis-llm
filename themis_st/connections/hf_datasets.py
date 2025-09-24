from typing import Final

import requests

from streamlit import logger
from streamlit.connections import BaseConnection

_LOGGER: Final = logger.get_logger(__name__)


class HFDatasetConnection(BaseConnection["HFDatasetConnection"]):
    """Connection to the Hugging Face Datasets API."""

    BASE_URL = "https://datasets-server.huggingface.co"

    def _connect(self, **kwargs) -> "HFDatasetConnection":
        self._token = self._secrets.get("token", None)
        if not self._token:
            _LOGGER.warning("No token found in `secrets.toml`. Access to gated datasets may be limited.")
        return self

    @property
    def token(self) -> str | None:
        return self._token

    @token.setter
    def token(self, value: str):
        _LOGGER.info("Setting token for HFDatasetConnection.")
        self._token = value

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def is_valid(self, dataset: str) -> requests.Response:
        return requests.get(f"{self.BASE_URL}/is-valid?dataset={dataset}", headers=self.headers)

    def list_splits(self, dataset: str) -> requests.Response:
        return requests.get(f"{self.BASE_URL}/splits?dataset={dataset}", headers=self.headers)

    def get_info(self, dataset: str, config: str | None = None) -> requests.Response:
        url = f"{self.BASE_URL}/info?dataset={dataset}"
        if config:
            url += f"&config={config}"
        return requests.get(url, headers=self.headers)

    def preview(self, dataset: str, config: str, split: str) -> requests.Response:
        return requests.get(
            f"{self.BASE_URL}/first-rows?dataset={dataset}&config={config}&split={split}", headers=self.headers
        )

    def get_rows(self, dataset: str, config: str, split: str, offset: int = 0, length: int = 100) -> requests.Response:
        return requests.get(
            f"{self.BASE_URL}/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}",
            headers=self.headers,
        )

    def search(
        self, dataset: str, config: str, split: str, query: str, offset: int = 0, length: int = 100
    ) -> requests.Response:
        return requests.get(
            f"{self.BASE_URL}/search?dataset={dataset}&config={config}&split={split}&query={query}&offset={offset}&length={length}",
            headers=self.headers,
        )

    def filter(
        self,
        dataset: str,
        config: str,
        split: str,
        where: str,
        orderby: str | None = None,
        offset: int = 0,
        length: int = 100,
    ) -> requests.Response:
        url = f"{self.BASE_URL}/filter?dataset={dataset}&config={config}&split={split}&where={where}&offset={offset}&length={length}"  # noqa: E501
        if orderby:
            url += f"&orderby={orderby}"
        return requests.get(url, headers=self.headers)

    def list_parquet(self, dataset: str) -> requests.Response:
        return requests.get(f"{self.BASE_URL}/parquet?dataset={dataset}", headers=self.headers)

    def get_size(self, dataset: str) -> requests.Response:
        return requests.get(f"{self.BASE_URL}/size?dataset={dataset}", headers=self.headers)

    def get_statistics(self, dataset: str, config: str, split: str) -> requests.Response:
        return requests.get(
            f"{self.BASE_URL}/statistics?dataset={dataset}&config={config}&split={split}", headers=self.headers
        )

    def get_croissant(self, dataset: str) -> requests.Response:
        return requests.get(f"{self.BASE_URL}/croissant?dataset={dataset}", headers=self.headers)
