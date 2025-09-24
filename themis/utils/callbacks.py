import re
import shutil
import logging

from typing import Any
from pathlib import Path

from omegaconf import DictConfig
from hydra.types import TaskFunction
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback

logger = logging.getLogger(__name__)


class MyCallback(Callback):
    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        pass

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        pass

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        pass

    def on_job_start(self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any) -> None:
        pass

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        output_dir = Path(config.hydra.runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if job_return.status == JobStatus.COMPLETED:
            self._clean_directory(output_dir=output_dir)
        elif job_return.status == JobStatus.FAILED:
            self._rmdir(output_dir=output_dir)

    def on_run_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        pass

    def _rmdir(self, output_dir: Path) -> None:
        contents = [f.name for f in output_dir.iterdir()]
        if contents == ["logs.log"]:
            self.log.info(f"Deleting empty directory: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            self.log.info(f"Non empty directory: {output_dir}")

    def _clean_directory(self, output_dir: Path) -> None:
        timestamp_pattern = re.compile(r"_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+(?=\.json)")

        files_with_timestamp = [f for f in output_dir.iterdir() if f.is_file() and timestamp_pattern.search(f.name)]

        for file in files_with_timestamp:
            new_name = timestamp_pattern.sub("", file.name)
            new_path = file.with_name(new_name)

            if new_path.exists():  # nt
                new_path.unlink()

            file.rename(new_path)
