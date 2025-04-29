from typing import Any

from rich import print as rich_print
from pydantic import Field, BaseModel, field_validator
from omegaconf import MISSING
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
    apply_chat_template: bool | None = Field(default=False)
    limit: float | int | None = Field(default=None)
    bootstrap_iters: int | None = Field(default=0, ge=0)
    random_seed: int = Field(default=2025)
    numpy_random_seed: int = Field(default=2025)
    torch_random_seed: int = Field(default=2025)
    fewshot_random_seed: int = Field(default=2025)
    gen_kwargs: GenerationArguments | None = Field(default=None)

    @field_validator("apply_chat_template")
    def validate_template(cls, value: bool | None) -> bool:
        if value is None:
            return False
        return value

    @field_validator("limit")
    def validate_limit(cls, value: float | int | None) -> float | int | None:
        if isinstance(value, float):
            if not (0 < value < 1):
                raise ValueError("If limit is a float, it must be between 0 and 1")
        return value


class ExperimentConfig(BaseModel, validate_assignment=True):
    model: str = Field(frozen=True)
    task: str | list = Field(...)
    eval_kwargs: EvalArguments = Field(frozen=True, default_factory=EvalArguments)

    @field_validator("task")
    def validate_task(cls, value: str | list) -> str | list:
        if value == MISSING:
            raise ValueError("Missing mandatory value: task")
        return value


class InterfaceConfig(BaseModel, frozen=True):
    name: str = Field(...)
    args: dict[str, Any] = Field(default_factory=dict)


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
