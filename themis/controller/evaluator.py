import logging

import lm_eval
from lm_eval.tasks import TaskManager

from themis.controller.indexer import experiment_exists
from themis.controller.logger import log_results
from themis.definitions.constants import TASK_PATH
from themis.model.model import get_generation_args, get_model
from themis.utils.config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Evaluation:
    """Holds parameters for lm_eval.simple_evaluate"""

    def __init__(self, cfg: Config) -> None:
        self.model = get_model(cfg=cfg)
        self.params = {TaskManager(include_path=TASK_PATH)}

        # Random seeds
        self.random_seed = cfg.seed
        self.numpy_random_seed = cfg.seed
        self.torch_random_seed = cfg.seed
        self.fewshot_random_seed = cfg.seed

        self.limit = cfg.limit  # limit for testing


class CrowsPairsMC(Evaluation):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg=cfg)
        self.tasks = [
            "crows_pairs_mc_age",
            "crows_pairs_mc_disability",
            "crows_pairs_mc_gender",
            "crows_pairs_mc_nationality",
            "crows_pairs_mc_orientation",
            "crows_pairs_mc_physical_appearance",
            "crows_pairs_mc_race_color",
            "crows_pairs_mc_religion",
            "crows_pairs_mc_socioeconomic",
        ]


class CrowsPairsQA(Evaluation):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg=cfg)

        self.tasks = [
            "crows_pairs_qa_age",
            "crows_pairs_qa_disability",
            "crows_pairs_qa_gender",
            "crows_pairs_qa_nationality",
            "crows_pairs_qa_orientation",
            "crows_pairs_qa_physical_appearance",
            "crows_pairs_qa_race_color",
            "crows_pairs_qa_religion",
            "crows_pairs_qa_socioeconomic",
        ]

        self.system_instruction = "You are an AI assistant tasked to answer multiple choice questions."
        self.apply_chat_template = True

        self.gen_kwargs = get_generation_args(cfg=cfg)


def eval_factory(cfg: Config) -> Evaluation:
    logger.info("Parsing evaluation parameters")

    if cfg.task == "crows_pairs_mc":
        return CrowsPairsMC(cfg=cfg)

    if cfg.task == "crows_pairs_qa":
        return CrowsPairsQA(cfg=cfg)

    raise NotImplementedError(f"{cfg.task} not implemented")


def evaluate(cfg: Config) -> None:
    eval_parameters = eval_factory(cfg=cfg)
    eval_args = vars(eval_parameters)

    logger.info("Running evaluation task")
    results = lm_eval.simple_evaluate(**eval_args)
    log_results(cfg=cfg, results=results)


def make_experiment(cfg: Config) -> None:
    experiment = experiment_exists(cfg=cfg)
    if experiment:
        logger.info(f"Experiment already run for given configuration\n{cfg}")
        # logger.info(cfg)
    else:
        evaluate(cfg=cfg)
