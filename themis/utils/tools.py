import re
import logging

import hydra
import coolname

from yaml import FullLoader, load
from lm_eval.tasks import TaskManager
from hydra.core.hydra_config import DictConfig

from themis.definitions.constants import TASK_PATH

logger = logging.getLogger(__name__)


# https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_colorlog/hydra_plugins/hydra_colorlog/conf/hydra/hydra_logging/colorlog.yaml
class CustomFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
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


def recompose_config(config_dir: str, overrides_path: list[str] | None = None) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return hydra.compose(config_name="config", return_hydra_config=False)


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, encoding="utf8") as yaml_fh:
        config = load(yaml_fh, Loader=FullLoader)

    return config


def slug(count: int) -> str:
    return coolname.generate_slug(count).replace("-", "_")


def to_string(args_dict: dict) -> str:
    return ",".join([f"{k}={v}" for k, v in args_dict.items()])


def sanitize_model_name(name: str) -> str:
    _, name = name.split("/")  # meta-llama / Llama-3.2-3B
    return re.sub(r"[^a-zA-Z0-9]", "_", name).lower()  # llama_3_2_3b


def sanitize_task_name(tasks: str | list) -> str:
    tasks = ",".join(tasks) if isinstance(tasks, list) else tasks
    return re.sub(r"\W", "_", tasks)


def format_size(num: int) -> str:
    """Format size in bytes into a human-readable string.

    Taken from https://stackoverflow.com/a/1094933
    """
    num_f = float(num)
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num_f) < 1000.0:
            return f"{num_f:3.1f}{unit}"
        num_f /= 1000.0
    return f"{num_f:.1f}Y"


def list_tasks() -> None:
    task_manager = TaskManager(include_path=TASK_PATH, include_defaults=False)
    print(task_manager.list_all_tasks())
