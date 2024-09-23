import logging
import os
import os.path as osp
import pickle
import random
import shutil
from datetime import datetime

from coolname import generate_slug
from lm_eval.loggers import WandbLogger

from themis.controller.indexer import experiment_exists, load_results
from themis.definitions.constants import (
    CONFIG_PATH,
    DATA_PATH,
    EXPERIMENTS_PATH,
    PROJECT,
)
from themis.utils.config import Config

logger = logging.getLogger(__name__)


def make_exp_dir() -> str:
    random.seed(os.urandom(128))
    save_path = osp.join(EXPERIMENTS_PATH, generate_slug())
    os.makedirs(save_path)
    return save_path


# def make_exp_dir(cfg: Config, results: dict) -> str:
#     model = osp.basename(cfg.model).lower()

#     date = datetime.fromtimestamp(results["date"])
#     date = date.strftime("%d%m%Y_%H%M%S")

#     save_path = osp.join(EXPERIMENTS_PATH, cfg.task, model, date)
#     os.makedirs(save_path, exist_ok=True)

#     return save_path


def dump_results(save_path: str, results: dict) -> None:
    f_path = osp.join(save_path, "results.pickle")
    with open(f_path, "wb") as handle:
        pickle.dump(results, handle)


def log_results(cfg: Config, results: dict) -> None:
    save_path = make_exp_dir()  # create unique path
    dump_results(save_path=save_path, results=results)  # save results dictionary
    shutil.copy(CONFIG_PATH, save_path)  # copy experiment config

    logger.info("Saving evaluation results and configuration")


def log_to_wandb(cfg: Config) -> None:
    results_path = experiment_exists(cfg=cfg)

    if results_path:
        results = load_results(f_path=results_path)

        name = f"{osp.basename(cfg.model).lower()}-{cfg.task}"

        wandb_logger = WandbLogger(
            name=name, project=PROJECT, job_type="eval", group=cfg.task, config=cfg, dir=DATA_PATH
        )
        logger.info("Logging evaluation results to Weights & Biases")
        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        wandb_logger.log_eval_samples(results["samples"])
    else:
        logger.info("No results found for given evaluation")
