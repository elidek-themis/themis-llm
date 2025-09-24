import json

from typing import Any
from pathlib import Path
from functools import partial

import numpy as np

from datasets import Dataset

from themis.definitions.constants import RAW_PATH


def process_results(doc, results):
    results, _ = zip(*results)  # remove is_greedy
    no_choices = int(len(results) // 2)

    return {
        "democratic": dict(zip(doc["choices"]["pro"], results[:no_choices])),
        "republican": dict(zip(doc["choices"]["contra"], results[no_choices:])),
    }


def norm_prob_diff(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, float]]:
    alias = doc["choices"]["alias"]
    no_choices = len(alias)
    results, _ = zip(*results)  # remove is_greedy

    # negative log likelihoods
    lls = np.array(results)
    lls_a, lls_b = lls[:no_choices], lls[no_choices:]

    # exponentiation to get probabilities
    e_a, e_b = np.exp(lls_a), np.exp(lls_b)
    total = e_a + e_b

    # normalize probabilities
    prob_a, prob_b = e_a / total, e_b / total

    # normalized probability differences
    diff = (prob_a - prob_b).tolist()

    return {"norm_prob_diff": dict(zip(alias, diff))}


def bypass(arr):
    return arr


def load_prompts(f_name: str, key: str, **kwargs) -> dict[str, Dataset]:
    template = kwargs.pop("template", None)
    choices = kwargs.pop("choices", None)
    assert template and choices

    f_path = Path(RAW_PATH) / "election" / f_name
    docs = json.load(f_path)

    for record in docs:
        record["template"] = template.format(state=record[key])

    dataset = Dataset.from_list(docs)
    dataset = dataset.add_column("choices", [choices] * len(dataset))
    dataset.choices = choices

    return {"train": dataset}


load_state_prompts = partial(load_prompts, f_name="states.json", key="state")
load_demographic_prompts = partial(load_prompts, f_name="demographics.json", key="persona")


# def load_state_prompts(**kwargs):
#     return load_prompts(f_name="states.json", key="state", **kwargs)


# def load_demographic_prompts(**kwargs):
#     return load_prompts(f_name="demographics.json", key="persona", **kwargs)
