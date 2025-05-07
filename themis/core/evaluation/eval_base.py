import abc

from lm_eval.api.model import TemplateLM

from themis.data.repository import Repository


class Evaluation(abc.ABC):
    @abc.abstractmethod
    def validate(self, repo: Repository) -> None:
        """Performs validations on the evaluation configuration.
        Args:
            repo (Repository): The repository instance to validate against.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, lm: TemplateLM) -> dict:
        """Performs evaluation using the underlying framework.
        Args:
            lm: The language model to evaluate.
        Returns:
            dict: The evaluation results.
        """
        pass
