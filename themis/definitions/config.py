import os
import json

from logging import getLogger
from pathlib import Path
from functools import cached_property
from dataclasses import field, asdict, dataclass

from rich import print as rich_print
from rich.panel import Panel
from rich.pretty import Pretty
from lm_eval.tasks import TaskManager
from lm_eval.utils import simple_parse_args_string
from lm_eval.evaluator import request_caching_arg_to_dict

logger = getLogger(__name__)


@dataclass
class LMEvalCLI:
    """Configuration for lm-evaluation-harness CLI.

    Validates and prepares configuration for running evaluations using
    lm_eval.evaluator.simple_evaluate().
    """

    model: str = "hf"
    tasks: str | None = None
    model_args: str | dict = field(default_factory=dict)
    num_fewshot: int | None = None
    batch_size: str = "1"
    max_batch_size: int | None = None
    device: str | None = None
    output_path: str | None = None
    limit: float | None = None
    samples: str | None = None
    use_cache: str | None = None
    cache_requests: str | None = None
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = False
    system_instruction: str | None = None
    apply_chat_template: bool | str = False
    fewshot_as_multiturn: bool = False
    show_config: bool = False
    include_path: str | None = None
    gen_kwargs: dict | None = None
    verbosity: str | None = None
    wandb_args: str = ""
    wandb_config_args: str = ""
    hf_hub_log_args: str = ""
    predict_only: bool = False
    seed: int | list | None = 2025
    trust_remote_code: bool = False
    confirm_run_unsafe_code: bool = False
    metadata: dict | None = None

    def __post_init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self._prep_hf_hub_args()
        self._validate_outputs()
        self._validate_chat_template()
        self._prep_metadata()
        self._validate_samples()
        self._validate_tasks()
        self._prep_trust_remote_code()

    def _prep_hf_hub_args(self):
        if self.output_path:
            self.hf_hub_log_args += f",output_path={self.output_path}"

        if os.environ.get("HF_TOKEN"):
            self.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"

    def _validate_outputs(self) -> None:
        if self.predict_only:
            self.log_samples = True

        if (self.log_samples or self.predict_only) and not self.output_path:
            raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    def _validate_chat_template(self):
        if self.fewshot_as_multiturn and self.apply_chat_template is False:
            raise ValueError(
                "When `fewshot_as_multiturn` is selected, `apply_chat_template` "
                "must be set (either to `True` or to the chosen template name)."
            )

    def _prep_metadata(self):
        # extract metadata from model_args
        model_args_metadata = (
            simple_parse_args_string(self.model_args)
            if isinstance(self.model_args, str)
            else self.model_args
            if isinstance(self.model_args, dict)
            else {}
        )

        # merge with explicit metadata
        explicit_metadata = (
            self.metadata
            if isinstance(self.metadata, dict)
            else simple_parse_args_string(self.metadata)
            if self.metadata
            else {}
        )

        self.metadata = model_args_metadata | explicit_metadata

    def _validate_samples(self):
        if self.limit:
            logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

        if self.samples:
            assert self.limit is None, "If --samples is not None, then --limit must be None."
            if (samples := Path(self.samples)).is_file():
                self.samples = json.loads(samples.read_text())
            else:
                self.samples = json.loads(self.samples)

    def _validate_tasks(self):
        if self.tasks is None:
            raise ValueError("Need to specify task to evaluate.")

    def _prep_trust_remote_code(self):
        if not self.trust_remote_code:
            return

        import datasets  # noqa: PLC0415

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        if isinstance(self.model_args, dict):
            self.model_args["trust_remote_code"] = True
        else:
            self.model_args = (
                f"{self.model_args},trust_remote_code=True" if self.model_args else "trust_remote_code=True"
            )

    @cached_property
    def evaluation_tracker_args(self):
        args = simple_parse_args_string(self.hf_hub_log_args)
        if "push_samples_to_hub" in args and not self.log_samples:
            logger.warning(
                "Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub."
            )

        return args

    @cached_property
    def task_manager(self) -> TaskManager:
        if self.include_path is not None:
            logger.info(f"Including path: {self.include_path}")

        return TaskManager(
            include_path=self.include_path,
            include_defaults=False,
            metadata=self.metadata,
        )

    @property
    def request_caching_args(self) -> dict:
        return request_caching_arg_to_dict(cache_requests=self.cache_requests)

    @property
    def seed_kwargs(self) -> dict:
        seed_names = ["random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed"]

        if isinstance(self.seed, int):
            return dict.fromkeys(seed_names, self.seed)
        elif isinstance(self.seed, list):
            assert len(self.seed) == len(seed_names), f"seed list must have 4 values, got {len(self.seed)}"
            return dict(zip(seed_names, self.seed))

        # Fallback
        return dict.fromkeys(seed_names, 2025)

    def get_simple_evaluate_kwargs(self) -> dict:
        tasks = self.tasks.split(",") if isinstance(self.tasks, str) else self.tasks

        return {
            "model": self.model,
            "model_args": self.model_args,
            "tasks": tasks,
            "task_manager": self.task_manager,
            "num_fewshot": self.num_fewshot,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "device": self.device,
            "use_cache": self.use_cache,
            "limit": self.limit,
            "samples": self.samples,
            "check_integrity": self.check_integrity,
            "write_out": self.write_out,
            "log_samples": self.log_samples,
            "system_instruction": self.system_instruction,
            "apply_chat_template": self.apply_chat_template,
            "fewshot_as_multiturn": self.fewshot_as_multiturn,
            "gen_kwargs": self.gen_kwargs,
            "predict_only": self.predict_only,
            "confirm_run_unsafe_code": self.confirm_run_unsafe_code,
            "metadata": self.metadata,
            **self.seed_kwargs,
            **self.request_caching_args,
        }

    def __str__(self) -> str:
        return json.dumps(asdict(self), indent=4)


def table_print(config: LMEvalCLI, title: str = "Valid configuration :heavy_check_mark:") -> None:
    """Prints the configuration in a rich panel.

    Args:
        config: The configuration to print.
        title: The title of the panel.
    """
    pretty = Pretty(config)
    panel = Panel(pretty, title=title, title_align="left")
    rich_print(panel)
