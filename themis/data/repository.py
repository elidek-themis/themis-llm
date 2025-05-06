import os
import pickle
import os.path as osp

from typing import Any
from dataclasses import dataclass

import pandas as pd
import evaluate

from tinydb import Query, TinyDB
from pydantic_yaml import parse_yaml_file_as
from huggingface_hub import scan_cache_dir
from tinydb.storages import MemoryStorage
from lm_eval.api.registry import METRIC_REGISTRY

from themis.utils.tools import format_size
from themis.definitions.config import ExperimentConfig
from themis.definitions.constants import EXPERIMENTS_PATH


@dataclass
class ExperimentDirectory:
    log: str = "__main__.log"
    config: str = "experiment.yaml"
    job_return: str = "job_return.pickle"

    files = [job_return, log, config]


@dataclass
class ExperimentOutput:
    root: str
    log: str
    config: ExperimentConfig
    results: dict[str, Any]

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def tasks(self) -> list:
        return self.config.task

    @property
    def task_configs(self) -> dict:
        return self.results.get("configs", {})

    @property
    def alias(self) -> str:
        _, alias = self.root.rsplit("/", 1)
        return alias

    def __str__(self):
        return self.alias

    def __repr__(self):
        return self.alias


class Repository:
    def __init__(self):
        # TODO: persistent storage requires
        # to serialize the pickled results
        self.db = TinyDB(storage=MemoryStorage)

        metrics = self._get_metrics()
        metrics_table = self.db.table("metrics")
        metrics_table.insert(metrics)

        model_hub = self._get_model_hub()
        model_hub_table = self.db.table("model_hub")
        for model in model_hub:
            model_hub_table.insert(model)

        experiments = self._get_experiments()
        experiments_table = self.db.table("experiments")
        for experiment in experiments:
            experiments_table.insert(experiment)

    def _get_metrics(self) -> dict:
        lm_eval_metrics = list(METRIC_REGISTRY.keys())
        hf_metrics = evaluate.list_evaluation_modules()

        return {
            "lm_eval_metrics": lm_eval_metrics,
            "hf_metrics": hf_metrics,
        }

    def _get_model_hub(self) -> dict:
        to_drop = ["repo_path", "revisions", "last_accessed", "last_modified", "nb_files"]

        cache_dir = scan_cache_dir()
        hf_repo = pd.DataFrame(cache_dir.repos).drop(columns=to_drop)
        hf_repo = hf_repo[hf_repo.repo_type == "model"]
        hf_repo.sort_values(by="size_on_disk", inplace=True)
        hf_repo["size_on_disk"] = hf_repo["size_on_disk"].map(format_size)

        hf_repo.rename(columns={"size_on_disk": "size", "repo_id": "model"}, inplace=True)
        hf_repo.drop("repo_type", axis=1, inplace=True)
        hf_repo.reset_index(drop=True, inplace=True)

        return hf_repo.to_dict(orient="records")

    def _get_experiments(self) -> list[dict]:
        experiments = []
        for root, _, files in os.walk(EXPERIMENTS_PATH):
            print(files)
            if files == ExperimentDirectory.files:
                log, config, results = self._parse_experiment_directory(path=root)

                experiments.append(
                    {
                        "dir": root,
                        "config": config,
                        "log": log,
                        "results": results,
                    }
                )

        return experiments

    def _parse_experiment_directory(self, path: str) -> tuple[str, dict, dict]:
        log_path = osp.join(path, ExperimentDirectory.log)
        with open(log_path) as file:
            log = file.read()

        config_path = osp.join(path, ExperimentDirectory.config)
        config = parse_yaml_file_as(model_type=ExperimentConfig, file=config_path).model_dump()

        results_path = osp.join(path, ExperimentDirectory.job_return)
        with open(results_path, "rb") as handle:
            results = pickle.load(handle)

        return log, config, results

    def load(self, config: ExperimentConfig) -> list[dict]:
        Experiment = Query()
        experiment_table = self.db.table("experiments")

        model = config.model
        task = config.task
        eval_kwargs = config.eval_kwargs.model_dump()

        results = experiment_table.search(
            (Experiment.config.model == model)
            & (Experiment.config.task.all(task))
            & (Experiment.config.eval_kwargs == eval_kwargs)
        )

        for result in results:
            print(f"Found experiment in {result['dir']}")

        return results

    def contains(self, config: ExperimentConfig, task: str | list) -> bool:
        Experiment = Query()
        experiment_table = self.db.table("experiments")

        model = config.model
        eval_kwargs = config.eval_kwargs.model_dump()

        if isinstance(task, str):
            task = [task]

        return experiment_table.contains(
            (Experiment.config.model == model)
            & (Experiment.config.task.all(task))
            & (Experiment.config.eval_kwargs == eval_kwargs)
        )
