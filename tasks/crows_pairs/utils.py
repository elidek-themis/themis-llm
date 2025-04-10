import datasets
from lm_eval.api.task import ConfigurableTask

def doc_to_choice(doc):
    return [doc["sent_more"], doc["sent_less"]]


def process_results(_, results):
    (l1, l2), _ = zip(*results)
    # if stereotypical sentence more likely (loglikelihood higher)
    # then treat this as predicting stereotyped sentence
    return {
        "likelihood1": l1,
        "likelihood2": l2,
        "likelihood_diff": abs(l1 - l2),
        "is_stereo": 1.0 if l1 > l2 else 0.0
    }

def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        return {"choices": (doc["sent_more"], doc["sent_less"])}

    return (
        dataset
        .map(_process_doc)
        .remove_columns(["sent_more", "sent_less"])
    )