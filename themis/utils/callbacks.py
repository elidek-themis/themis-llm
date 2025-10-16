import json
import shutil
import logging

from typing import Any
from pathlib import Path

from omegaconf import DictConfig
from hydra.types import TaskFunction
from hydra.utils import instantiate
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback

from themis.definitions.config import LMEvalCLI, table_print

logger = logging.getLogger(__name__)


class MyCallback(Callback):
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.eval_config: LMEvalCLI | None = None

    def on_job_start(self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any) -> None:
        self.eval_config = instantiate(config.evaluation)
        table_print(config=self.eval_config)
        self.logger.info("Configuration validated and stored in callback")

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        output_path = Path(self.eval_config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if job_return.status == JobStatus.COMPLETED:
            self._save_results(results=job_return.return_value, output_path=output_path)
        elif job_return.status == JobStatus.FAILED:
            self._rm_dir(output_dir=output_path.parent)

    def _save_results(self, results: dict[str, Any], output_path: Path) -> None:
        """Save evaluation results to JSON file."""
        try:
            # pop samples dict (task_name -> list of samples)
            samples_dict = results.pop("samples", None)

            # save main results
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"✓ Saved results to: {output_path}")

            # save samples per task as JSONL
            if samples_dict and self.eval_config.log_samples:
                output_dir = output_path.parent

                for task_name, task_samples in samples_dict.items():
                    self._save_task_samples(
                        task_name=task_name,
                        samples=task_samples,
                        output_dir=output_dir,
                    )

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}", exc_info=True)

    def _save_task_samples(self, task_name: str, samples: list[dict], output_dir: Path) -> None:
        """Save per-task samples as JSONL file."""
        try:
            samples_path = output_dir / f"samples_{task_name}.jsonl"

            with samples_path.open("w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")

            self.logger.info(f"✓ Saved {len(samples)} samples to: {samples_path}")

        except Exception as e:
            self.logger.error(f"Failed to save samples for {task_name}: {e}", exc_info=True)

    def _rm_dir(self, output_dir: Path) -> None:
        """Remove output directory if it only contains logs."""

        def is_empty(d: Path) -> bool:
            contents = [f.name for f in d.iterdir()]
            return contents == ["logs.log"]

        if is_empty(output_dir):
            self.logger.info(f"Deleting empty/failed directory: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            self.logger.info(f"Keeping non-empty directory: {output_dir}")
