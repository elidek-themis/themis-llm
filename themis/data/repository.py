import os
import copy
import json

from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from lm_eval.loggers.utils import remove_none_pattern

from themis.definitions.constants import EXPERIMENTS_PATH


@dataclass
class ExperimentFolder:
    path: Path
    files: list[str]

    def __post_init__(self):
        with open(self.path / "logs.log") as f:
            self.log = f.read()

        with open(self.path / "results.json") as f:
            self.results = json.load(f)
        self._calc_metrics()

        self._samples = {}
        for f in self.files:
            if f.startswith("samples_") and f.endswith(".jsonl"):
                key = f[len("samples_") : -len(".jsonl")]
                self._samples[key] = self.path / f

    def samples(self, key: str):
        if key not in self._samples:
            raise KeyError(f"Sample key '{key}' not found.")
        file_path = self._samples[key]
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def _calc_metrics(self) -> None:  # fix subtasks ,none
        metrics = copy.deepcopy(self.results.get("results", {}))

        tmp_metrics = copy.deepcopy(metrics)
        for task_name in self.tasks:
            task_metrics = tmp_metrics.get(task_name, {})
            for metric_name, metric_value in task_metrics.items():
                _metric_name, removed = remove_none_pattern(metric_name)
                if isinstance(metric_value, str):
                    metrics[task_name].pop(metric_name)
                elif removed:
                    metrics[task_name][_metric_name] = metric_value
                    metrics[task_name].pop(metric_name)

        self.metrics = metrics

    @property
    def config(self):
        return self.results.get("config")

    @property
    def task_configs(self):
        return self.results.get("configs")

    @property
    def model(self) -> str:
        model = self.results["model_name_sanitized"]
        # _, model = model.split("/")
        return model

    @property
    def tasks(self):
        return list(self.results.get("group_subtasks"))

    @classmethod
    def is_valid(cls, files: list[str]) -> bool:
        has_results = "results.json" in files
        has_log = "logs.log" in files
        has_samples = any(f.startswith("samples_") and f.endswith(".jsonl") for f in files)
        return has_results and has_log and has_samples

    @property
    def is_group(self) -> bool:
        return "group" in self.results

    def __str__(self) -> str:
        return json.dumps(self.results, indent=4)


def get_repo() -> pd.DataFrame:
    experiments = []
    for root, _, files in os.walk(EXPERIMENTS_PATH):
        if ExperimentFolder.is_valid(files=files):
            path = Path(root)
            experiments.append(ExperimentFolder(path=path, files=files))

    data = [(e.model, e.tasks, e) for e in experiments]
    runs = pd.DataFrame(data, columns=("model", "task", "data"))

    return runs.explode("task")
