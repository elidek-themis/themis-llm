import os
import os.path as osp
import pickle
from typing import Union

from themis.definitions.constants import EXPERIMENTS_PATH
from themis.utils.config import Config, load_config


def experiment_exists(cfg: Config) -> Union[str, bool]:
    """Checks if an experiment exists and returns its directory"""
    for root, _, files in os.walk(EXPERIMENTS_PATH):
        if files:  # if not empty directory
            results_path, config_path = files
            config = load_config(osp.join(root, config_path))

            if config == cfg:
                return osp.join(root, results_path)

    return False


def load_results(f_path: str) -> dict:
    with open(f_path, "rb") as handle:
        results = pickle.load(handle)
    return results


def load_task_results(task: str) -> dict:
    """Returns the results of all models for a given task"""
    experiments = {}
    experiment_path = osp.join(EXPERIMENTS_PATH, task)

    for root, _, files in os.walk(experiment_path):
        if files:
            results_path, config_path = files
            config = load_config(osp.join(root, config_path))

            experiments[config.model] = load_results(osp.join(root, results_path))

    return experiments
