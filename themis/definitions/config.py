from typing import Any, Dict, Optional, Union

from omegaconf import MISSING
from pydantic import BaseModel, Field, field_validator
from rich import print as rich_print
from rich.panel import Panel
from rich.pretty import Pretty


class GenerationArguments(BaseModel, frozen=True):
    temperature: float = Field(default=0.7, ge=0, le=1)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=50, ge=1, le=100)
    max_tokens: int = Field(default=100, ge=1)

    @field_validator("max_tokens")
    def validate_max_tokens(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_tokens must be at least 1")
        return value


class EvalArguments(BaseModel, frozen=True):
    apply_chat_template: Optional[bool] = Field(default=False)
    limit: Optional[Union[float, int]] = Field(default=None)
    bootstrap_iters: Optional[int] = Field(default=0, ge=0)
    random_seed: int = Field(default=2025)
    numpy_random_seed: int = Field(default=2025)
    torch_random_seed: int = Field(default=2025)
    fewshot_random_seed: int = Field(default=2025)
    gen_kwargs: Optional[GenerationArguments] = Field(default=None)

    @field_validator("apply_chat_template")
    def validate_template(cls, value: Optional[bool]) -> bool:
        if value is None:
            return False
        return value

    @field_validator("limit")
    def validate_limit(cls, value: Optional[Union[float, int]]) -> Optional[Union[float, int]]:
        if isinstance(value, float):
            if not (0 < value < 1):
                raise ValueError("If limit is a float, it must be between 0 and 1")
        return value


class ExperimentConfig(BaseModel, validate_assignment=True):
    model: str = Field(frozen=True)
    task: Union[str, list] = Field(...)
    eval_kwargs: EvalArguments = Field(frozen=True, default_factory=EvalArguments)

    @field_validator("task")
    def validate_task(cls, value: Union[str, list]) -> Union[str, list]:
        if value == MISSING:
            raise ValueError("Missing mandatory value: task")
        return value


class InterfaceConfig(BaseModel, frozen=True):
    name: str = Field(...)
    args: Dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel, frozen=True):
    model: str = Field(..., repr=False)
    seed: int = Field(..., repr=False)
    interface: InterfaceConfig = Field(...)
    experiment: ExperimentConfig = Field(...)

    @field_validator("model")
    def validate_model(cls, value: str) -> str:
        if value == MISSING:
            raise ValueError("Missing mandatory value: model")
        return value

    @field_validator("seed")
    def validate_seed(cls, value: int) -> int:
        if value == MISSING:
            raise ValueError("Missing mandatory value: seed")
        return value


def table_print(config: Config, title: str = "Valid configuration :heavy_check_mark:") -> None:
    """Prints the configuration in a rich panel.

    Args:
        config: The configuration to print.
        title: The title of the panel.
    """
    pretty = Pretty(config)
    panel = Panel(pretty, title=title, title_align="left")
    rich_print(panel)
