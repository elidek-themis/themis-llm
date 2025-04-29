import pickle
import shutil
import logging

from typing import Any
from pathlib import Path

import yaml

from omegaconf import DictConfig
from hydra.types import TaskFunction
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback

from themis.definitions.config import ExperimentConfig
from themis.definitions.exceptions import ExperimentExists


class MyCallback(Callback):
    output_dir: Path

    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_start(self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any) -> None:
        pass

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        output_dir = Path(config.hydra.runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir is not None

        try:
            return_value = job_return.return_value  # can raise
            if return_value:  # check JobReturn.status maybe
                self._save_results(config=config, results=return_value, output_dir=output_dir)
        # log the raised Exception when the JobStatus is not COMPLETED
        except ExperimentExists as e:
            self.log.error(e, exc_info=True)
            self._rm_dir(output_dir)
            return
        except Exception:
            import traceback

            print(traceback.format_exc())
            self.log.error(job_return.return_value, exc_info=True)
            self._rm_dir(output_dir)
        finally:
            job_return.status = JobStatus.COMPLETED

    def _rm_dir(self, output_dir: Path):
        self.log.info(f"Deleting empty dir {output_dir}")
        shutil.rmtree(output_dir)

    def _save_results(self, config: DictConfig, results: dict[str, Any], output_dir: Path) -> None:
        exp_cfg: ExperimentConfig = results.get("config", {})

        filename = "experiment.yaml"
        self.log.info(f"Saving experiment config in {output_dir / filename}")
        with open(str(output_dir / filename), "w") as file:
            yaml.dump(exp_cfg.model_dump(), file, default_flow_style=False)

        filename = "job_return.pickle"
        self.log.info(f"Saving job_return in {output_dir / filename}")
        with open(str(output_dir / filename), "wb") as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
