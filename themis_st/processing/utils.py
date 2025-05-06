import logging

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
