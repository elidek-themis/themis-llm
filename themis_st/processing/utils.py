import logging
import time

import pandas as pd
from huggingface_hub import scan_cache_dir
from pydantic import BaseModel, field_validator

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
    def validate_task(cls, v):
        if v not in ALL_TASKS:
            raise ValueError(f"task should be any of {ALL_TASKS}")
        return v


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


def load():
    logger.info("🏃 Running task ")
    time.sleep(2)
    logger.info("✅ Finished task")
    time.sleep(2)
    logger.info(" Exit")


def get_model_hub() -> pd.DataFrame:
    to_drop = ["repo_path", "revisions", "last_accessed", "last_modified", "nb_files"]

    cache_dir = scan_cache_dir()
    hf_repo = pd.DataFrame(cache_dir.repos).drop(columns=to_drop)
    hf_repo = hf_repo[hf_repo.repo_type == "model"]
    hf_repo.sort_values(by="size_on_disk", inplace=True)
    hf_repo["size_on_disk"] = hf_repo["size_on_disk"].map(format_size)

    hf_repo.rename(columns={"size_on_disk": "size", "repo_id": "model"}, inplace=True)
    hf_repo.drop("repo_type", axis=1, inplace=True)
    hf_repo.reset_index(drop=True, inplace=True)

    return hf_repo
