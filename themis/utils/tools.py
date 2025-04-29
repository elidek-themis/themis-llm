import re
import logging

import hydra
import coolname

from yaml import FullLoader, load
from hydra.core.hydra_config import DictConfig

logger = logging.getLogger(__name__)


class CustomFormatter(logging.Formatter):
    def format(self, record):
        if "lm_eval" in record.pathname:
            module = "LM-Eval"
        elif "hydra" in record.pathname:
            module = "HYDRA"
        elif "vllm" in record.pathname:
            module = "vLLM"
        elif "themis" in record.pathname:
            module = "Themis"

        return "[{} - {}] [{} {}:{}] {}".format(
            module,
            record.levelname,
            self.formatTime(record, datefmt="%m-%d %H:%M:%S"),
            record.filename,
            record.lineno,
            record.getMessage(),
        )


def model_name(name: str) -> str:
    _, name = name.split("/")  # meta-llama / Llama-3.2-3B
    return re.sub(r"[^a-zA-Z0-9]", "_", name).lower()  # llama_3_2_3b


def slug(count: int) -> str:
    return coolname.generate_slug(count).replace("-", "_")


def recompose_config(config_dir: str, overrides_path: list[str] | None = None) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return hydra.compose(config_name="config", return_hydra_config=False)


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, encoding="utf8") as yaml_fh:
        config = load(yaml_fh, Loader=FullLoader)

    return config
