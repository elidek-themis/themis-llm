import os.path as osp
from logging import config as logging_config

from yaml import FullLoader, dump, load

from themis.definitions.constants import CONFIG_PATH, TEMPLATES_PATH


def load_prompt_template(name: str) -> str:
    prompt_templates = load_yaml_config(osp.join(TEMPLATES_PATH, "prompt_templates.yaml"))

    try:
        return prompt_templates[name]
    except KeyError:
        # logger.info("Unknown template name, using empty prompt template")
        return prompt_templates["empty_template"]


def make_active_prompt_template(name: str):
    active_template_path = osp.join(CONFIG_PATH, "active_template.yaml")

    template = load_prompt_template(name=name)
    with open(active_template_path, "w") as f:
        dump({"doc_to_text": template}, stream=f)


def load_yaml_config(config_path: str) -> dict:
    with open(config_path) as yaml_fh:
        config = load(yaml_fh, Loader=FullLoader)

    return config


def initialize_logging(config_path: str) -> None:
    """
    Setup logging according to the configuration in the given file.
    :param str config_path: The path to the file containing the logging configuration
    :return:
    """
    config_description = load_yaml_config(config_path=config_path)
    logging_config.dictConfig(config_description)
