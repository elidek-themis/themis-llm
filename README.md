### Installation
Use `make install` in order to reproduce the environment. \
Alternatively, use `make install-dev` in order to install in development mode.

### LM Evaluation Harness
The backbone of the experiments is based on `lm-eval-harness`. \
Tasks are described through `yaml` configuration files. \
[Hugging Face Builders](https://huggingface.co/docs/datasets/package_reference/builder_classes) \
[Create a new task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md)

Builder classes and tasks are listed in `themis-llm\themis\data\builders` and `themis-llm\tasks` respectively.

### Configuration
We use Hydra for our configuration management. \
The main experiment config is described in `themis-llm\data\conf\config.yaml`

### Example
1) Create a new Huggingface Builder and lm-eval task and populate the directories.
2) Update the HydraConfig
   - mandatory values are `task` and `model`
   - `task` can be either a single string or a list of strings
   - use `poetry run themis` to run the experiment
     - overrides are supported through `poetry run themis +task=task_1` | `poetry run themis +task=[task_2,task_3]`
     - multiruns are supported through `poetry run themis -m model=model_1,model_2`
4) Experiments are saved with a three letter slug, e.g. `themis-llm\data\experiments\tangerine_chinchilla_of_tenacity`, \
the directory contains
   - `__main__.log`, the logs of the experiment
   - `experiment.yaml`, a copy of the HydraConfig used for uniqueness
   - `job_return.pickle`, the results of the task, as returned by lm-eval-harness
