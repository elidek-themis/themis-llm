from dotenv import load_dotenv

from themis.definitions.constants import CONFIG_PATH, LOG_CONFIG_PATH
from themis.utils.config import Config, load_config, load_log_config


def init() -> Config:
    _ = load_dotenv()
    load_log_config(LOG_CONFIG_PATH)
    cfg = load_config(CONFIG_PATH)

    return cfg
