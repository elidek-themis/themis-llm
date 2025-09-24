from collections.abc import Callable

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance


class MultipleChoice(ConfigurableTask):
    OUTPUT_TYPE = "multiple_choice"
    VERSION = 0

    def __init__(self, config) -> None:
        config.pop("class", None)
        config.update({"output_type": self.OUTPUT_TYPE})
        # config.update({"aggregate_metric": self.aggregation})
        config.update({"metadata": {"version": self.VERSION}})
        super().__init__(config=config)

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> list[Instance] | Instance:
        apply_chat_template = kwargs.pop("apply_chat_template", False)
        chat_template: Callable | None = kwargs.pop("chat_template", None)

        choices = self.doc_to_choice(doc)
        target_delimiter = self.config.target_delimiter
        # if apply_chat_template:
        #     target_delimiter = ""
        arguments = [(ctx, f"{target_delimiter}{cont}") for cont in choices]

        request_list = [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=arg,
                idx=i,
                **kwargs,
            )
            for i, arg in enumerate(arguments)
        ]

        return request_list
