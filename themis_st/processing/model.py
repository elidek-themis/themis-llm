import logging
import time

import requests
from openai import OpenAI

from themis.definitions.config import InterfaceConfig

logger = logging.getLogger(__name__)


class Connection:
    def __init__(self, url, api_key):
        self.credentials = {"url": url, "api_key": api_key}

        self.client = OpenAI(
            api_key=api_key,
            base_url=url + "/v1",
        )

    @property
    def url(self):
        return self.credentials["url"]

    @property
    def api_key(self):
        return self.credentials["api_key"]

    @staticmethod
    def request(method, url, **kwargs):
        response = requests.request(method, url, **kwargs)
        return response
