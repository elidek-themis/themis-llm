from pathlib import Path

import pandas as pd
import datasets

from themis.definitions.constants import RAW_PATH

from .utils import get_differences

_CITATION = """\
@inproceedings{neveol2022french,
  title={French CrowS-Pairs: Extending a challenge dataset for measuring social bias in masked language models to a language other than English},
  author={N{\'e}v{\'e}ol, Aur{\'e}lie and Dupont, Yoann and Bezan{\\c{c}}on, Julien and Fort, Kar{\"e}n},
  booktitle={ACL 2022-60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
"""  # noqa: E501

_DESCRIPTION = """Corrections over the english revised version of CrowS-Pairs"""

_HOMEPAGE = "https://gitlab.inria.fr/french-crows-pairs/acl-2022-paper-data-and-code/-/tree/main"

_LICENSE = ""

_URLS = None


class CrowsPairsConfig(datasets.BuilderConfig):
    def __init__(
        self,
        mask_token: str,
        min_mask_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_token = mask_token
        self.min_mask_size = min_mask_size


class CrowsPairsPrompts(datasets.GeneratorBasedBuilder):
    "CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models"

    VERSION = datasets.Version("1.2.0")

    BUILDER_CONFIGS = [
        CrowsPairsConfig(
            name="english",
            mask_token="<MASK>",
            min_mask_size=50,
            version=VERSION,
            description="English CrowS-Pairs",
        )
    ]

    DEFAULT_CONFIG_NAME = "english"

    def _info(self):
        features = datasets.Features(
            {
                "template": datasets.Value("string"),
                "sent_more": datasets.Value("string"),
                "sent_less": datasets.Value("string"),
                "stereo_antistereo": datasets.Value("string"),
                "bias_type": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        file_path = Path(RAW_PATH) / "crows_pairs" / "crows_pairs_EN_revised+210_corrections.csv"
        data_file = dl_manager.download_and_extract(file_path)

        return [
            datasets.SplitGenerator(
                name="default",
                gen_kwargs={"filepath": data_file, "split": "default"},
            ),
            datasets.SplitGenerator(
                name="mask",
                gen_kwargs={"filepath": data_file, "split": "mask"},
            ),
        ]

    def _generate_examples(self, filepath: str, split: str):
        def skip_condition(affix, min_mask_size):
            is_min_len = any(len(infix.split()) > min_mask_size for infix in (affix.infix_1, affix.infix_2))
            is_empty = not all((affix.infix_1, affix.infix_2))
            return is_min_len or is_empty

        df = pd.read_csv(filepath, sep="\t", index_col=0)

        if split == "default":
            for key, row in enumerate(df.to_dict(orient="records")):
                yield key, {"template": ""} | row
        elif split == "mask":
            mask = self.config.mask_token
            min_mask_size = self.config.min_mask_size

            for key, row in enumerate(df.to_dict(orient="records")):
                affix = get_differences(sent_1=row["sent_more"], sent_2=row["sent_less"])
                if skip_condition(affix, min_mask_size):
                    continue  # noqa: E701
                yield (
                    key,
                    {
                        "template": affix.to_template(mask_token=mask)["template"],
                        "sent_more": affix.infix_1,
                        "sent_less": affix.infix_2,
                        "stereo_antistereo": row["stereo_antistereo"],
                        "bias_type": row["bias_type"],
                    },
                )
