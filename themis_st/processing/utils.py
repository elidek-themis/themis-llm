import logging

from typing import Any
from dataclasses import dataclass

import pandas as pd

from pydantic import BaseModel, field_validator

from themis.definitions.config import ExperimentConfig

logger = logging.getLogger(__name__)

ALL_TASKS = ["auto", "generate", "embedding", "embed", "classify", "score", "reward", "transcription"]


class VLLMArgs(BaseModel):
    model: str
    task: str = "generate"
    max_model_len: int
    max_logprobs: int
    swap_space: int
    cpu_offload_gb: float
    gpu_memory_utilization: float
    seed: int

    @field_validator("task")
    def validate_task(cls, v: str) -> str:
        if v not in ALL_TASKS:
            raise ValueError(f"task should be any of {ALL_TASKS}")
        return v


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
    def tasks(self) -> str | list[str]:
        return self.config.task

    @property
    def task_configs(self) -> dict[str, Any]:
        return self.results.get("configs", {})

    @property
    def alias(self) -> str:
        _, alias = self.root.rsplit("/", 1)
        return alias

    def __str__(self) -> str:
        return self.alias

    def __repr__(self) -> str:
        return self.alias


def get_runs_df(runs: list) -> pd.DataFrame:
    columns = ("model", "task", "output")

    runs = [
        ExperimentOutput(
            root=run["dir"],
            log=run["log"],
            config=ExperimentConfig(**run["config"]),
            results=run["results"],
        )
        for run in runs
    ]

    data = [(e.model, e.tasks, e) for e in runs]
    runs_df = pd.DataFrame(data, columns=columns)

    return runs_df.explode("task").reset_index(drop=True)
