from typing import Any

from rich import print as rich_print
from pydantic import Field, BaseModel, field_validator
from omegaconf import MISSING
from rich.panel import Panel
from rich.pretty import Pretty


class InterfaceConfig(BaseModel, frozen=True):
    # TODO: make more specific
    model: str
    model_args: str | dict[str, Any]


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


class LMEvalArguments(BaseModel, validate_assignment=True):
    model: str = Field(default="themis-singleton")
    model_args: InterfaceConfig = Field()
    tasks: str | list = Field()
    apply_chat_template: bool | None = Field(default=False)
    limit: float | int | None = Field(default=None)
    bootstrap_iters: int | None = Field(default=0, ge=0)
    random_seed: int = Field(default=2025)
    numpy_random_seed: int = Field(default=2025)
    torch_random_seed: int = Field(default=2025)
    fewshot_random_seed: int = Field(default=2025)
    use_cache: str = Field()
    gen_kwargs: GenerationArguments | None = Field(default=None)

    @field_validator("tasks")
    def validate_tasks(cls, value: str | list) -> str | list:
        if value == MISSING:
            raise ValueError("Missing mandatory value: tasks")
        return value

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

    @field_validator("random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed")
    def validate_seed(cls, value: int) -> int:
        if value == MISSING:
            raise ValueError("Missing mandatory value: seed")
        return value


def table_print(config: BaseModel, title: str = "Valid configuration :heavy_check_mark:") -> None:
    """Prints the configuration in a rich panel.

    Args:
        config: The configuration to print.
        title: The title of the panel.
    """
    pretty = Pretty(config)
    panel = Panel(pretty, title=title, title_align="left")
    rich_print(panel)
