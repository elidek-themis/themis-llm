import logging

from typing import Any

import requests

from openai import OpenAI

logger = logging.getLogger(__name__)


class Connection:
    def __init__(self, url: str, api_key: str) -> None:
        self.credentials = {"url": url, "api_key": api_key}

        self.client = OpenAI(
            api_key=api_key,
            base_url=url + "/v1",
        )

    @property
    def url(self) -> str:
        return self.credentials["url"]

    @property
    def api_key(self) -> str:
        return self.credentials["api_key"]

    @staticmethod
    def request(method: str, url: str, **kwargs: Any) -> requests.Response:
        response = requests.request(method, url, **kwargs)
        return response
