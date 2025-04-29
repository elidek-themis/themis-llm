import os
import pickle
import os.path as osp

from typing import Any
from dataclasses import dataclass

import pandas as pd

from pydantic_yaml import parse_yaml_file_as

from themis.definitions.config import ExperimentConfig
from themis.definitions.constants import EXPERIMENTS_PATH


@dataclass
class ExperimentDirectory:
    log: str = "__main__.log"
    config: str = "experiment.yaml"
    job_return: str = "job_return.pickle"

    @property
    def files(self):
        return [self.job_return, self.log, self.config]


@dataclass
class ExperimentOutput:
    root: str
    _results: dict[str, Any] = None

    def __post_init__(self):
        log_path = osp.join(self.root, ExperimentDirectory.log)
        with open(log_path) as file:
            self.log = file.read()

        config_path = osp.join(self.root, ExperimentDirectory.config)
        self.config = parse_yaml_file_as(model_type=ExperimentConfig, file=config_path)

        self.results_path = osp.join(self.root, ExperimentDirectory.job_return)

    @property
    def results(self) -> dict[str, Any]:
        if self._results is None:
            with open(self.results_path, "rb") as handle:
                self._results = pickle.load(handle)
        return self._results

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
    def __init__(self) -> None:
        columns = ("model", "task", "output")
        runs = self._read_experiments()

        data = [(e.model, e.tasks, e) for e in runs]
        runs = pd.DataFrame(data, columns=columns)

        self.runs: pd.DataFrame = runs.explode("task").reset_index(drop=True)

    def _read_experiments(self) -> list[ExperimentOutput]:
        experiment_directory = ExperimentDirectory()
        runs = []

        for root, _, files in os.walk(EXPERIMENTS_PATH):
            if files == experiment_directory.files:
                runs.append(ExperimentOutput(root=root))

        return runs

    def contains(self, config: ExperimentConfig, task: str) -> bool:
        runs = self.load(model=config.model, task=task)

        eval_kwargs = config.eval_kwargs
        if any(runs):
            run_kwargs = runs.output.map(lambda x: x.config.eval_kwargs)
            # ruff: noqa: C419
            return any([eval_kwargs == kwargs for kwargs in run_kwargs])

        return False

    def load(self, model: str | list | None = None, task: str | list | None = None) -> pd.DataFrame:
        runs = self.runs

        for key, value in {"model": model, "task": task}.items():
            if value is not None:
                in_ = value if isinstance(value, list) else [value]
                runs = runs[runs[key].isin(in_)]

        return runs

    def __repr__(self):
        return self.runs.to_string()
