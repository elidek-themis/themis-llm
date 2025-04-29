from collections.abc import Mapping, Callable, Iterable

import numpy as np

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

    def doc_to_text(self, doc, doc_to_text=None):
        return doc["input"]

    def doc_to_target(self, doc: Mapping, doc_to_target=None) -> int | str | list:
        pass

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset

    def fewshot_context(
        self,
        doc: str,
        num_fewshot: int,
        system_instruction: str | None = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Callable | None = None,
        gen_prefix: str | None = None,
    ) -> str:
        # if system_instruction:
        #     system_instruction = utils.apply_template(system_instruction, doc)
        # if gen_prefix:
        #     gen_prefix = utils.apply_template(gen_prefix, doc)
        # super().fewshot_context(
        #     doc=doc,
        #     num_fewshot=num_fewshot,
        #     system_instruction=system_instruction,
        #     apply_chat_template=apply_chat_template,
        #     fewshot_as_multiturn=fewshot_as_multiturn,
        #     chat_template=chat_template,
        #     gen_prefix=gen_prefix
        # )

        ctx = self.doc_to_text(doc)
        if isinstance(ctx, list):
            if apply_chat_template:
                chat_ctx = chat_template(ctx)
                return chat_ctx
            else:
                raise Exception("Got chat template format, but apply_chat_template is false.")
        else:
            return ctx

    def _create_arguments(self, doc: dict, ctx: str) -> list[tuple]:
        arguments = []

        for choice in doc["choices"]:
            arguments.append((ctx, f" {choice}"))  # whitespace in completion
            if "acc_mutual_info" in self._metric_fn_list.keys():
                arguments.append(("", f"{choice}"))  # unconditional loglikelihood

        return arguments

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> list[Instance]:
        kwargs.pop("apply_chat_template", False)
        kwargs.pop("chat_template", None)

        arguments = self._create_arguments(doc=doc, ctx=ctx)
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

    def process_results(self, doc: dict, results: Iterable[tuple[float, bool]]) -> dict:
        result_dict = {}
        lls, _ = zip(*results)

        choices = doc["choices"]

        if 2 * len(choices) == len(lls) and "acc_mutual_info" in self._metric_fn_list.keys():
            lls_unconditional = lls[1::2]  # unconditional answer loglikelihoods
            if len(lls_unconditional) != len(choices):
                raise ValueError
            lls = lls[::2]  # conditional loglikelihoods

            lls, lls_unconditional = map(np.array, (lls, lls_unconditional))
            result_dict.update({"acc": lls, "acc_mutual_info": lls - lls_unconditional})

        if "acc_norm" in self._metric_fn_list.keys():
            completion_len = np.array([float(len(i)) for i in choices])
            result_dict.update({"acc_norm": lls / completion_len})

        result_dict.update({"acc": np.array(lls)})

        return result_dict

    def aggregation(self):
        choices = self.config.dataset_kwargs["choices"]

        def y_map(y):
            return dict(zip(choices, y))

        byp = lambda _: _
        # agg = lambda _: list(map(y_map, _))

        return {"acc": byp, "acc_norm": byp, "acc_mutual_info": byp}
