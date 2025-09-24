import datasets


def process_results(doc, results):
    (l1, l2), _ = zip(*results)

    gold = doc["label"]
    argmax = 0 if l1 > l2 else 1

    return {
        "acc": 1 if gold == argmax else 0,
        "likelihood_diff": abs(l1 - l2),
    }


def filter_dataset(dataset: datasets.Dataset, gender: str) -> datasets.Dataset:
    return dataset.filter(lambda example: example["gender"] == gender)


def filter_male(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "male")


def filter_female(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "female")


def filter_neutral(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "neutral")
