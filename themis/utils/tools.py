from logging import config as logging_config

from yaml import FullLoader, load


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
