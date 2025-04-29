# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models"""

import pandas as pd
import datasets

_CITATION = """\
@inproceedings{nangia2020crows,
    title = "{CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models}",
    author = "Nangia, Nikita  and
      Vania, Clara  and
      Bhalerao, Rasika  and
      Bowman, Samuel R.",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics"
}
"""
# ruff: noqa: E501
_DESCRIPTION = """\
CrowS-Pairs, a challenge dataset for measuring the degree to which U.S. stereotypical biases present in the masked language models (MLMs).
"""


_URLS = [
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv",
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/refs/heads/master/data/prompts.csv",
]

_BIAS_TYPES = [
    "race-color",
    "socioeconomic",
    "gender",
    "disability",
    "nationality",
    "sexual-orientation",
    "physical-appearance",
    "religion",
    "age",
]


class CrowsPairsPrompts(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [datasets.BuilderConfig(name="test", version=VERSION, description="CrowS-Pairs")]

    DEFAULT_CONFIG_NAME = "test"

    def _info(self):
        features = datasets.Features(
            {
                "prompt": datasets.Value("string"),
                "sent_more": datasets.Value("string"),
                "sent_less": datasets.Value("string"),
                # "choices": datasets.Sequence(datasets.Value("string")),
                "bias_type": datasets.ClassLabel(names=_BIAS_TYPES),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepaths": data_files},
            )
        ]

    def _generate_examples(self, filepaths):
        cols = ["sent_more", "sent_less", "stereo_antistereo", "bias_type"]

        data_path, prompts_path = filepaths

        df = pd.read_csv(data_path, usecols=cols)
        prompts = pd.read_csv(prompts_path)["prompt"]
        df = pd.concat([df, prompts], axis=1)

        for key, row in enumerate(df.to_dict(orient="records")):
            # choices = (row["sent_more"], row["sent_less"])
            print(row["bias_type"])
            yield (
                key,
                {
                    "sent_more": row["sent_more"],
                    "sent_less": row["sent_less"],
                    "prompt": row["prompt"],
                    # "choices": choices,
                    "bias_type": row["bias_type"],
                },
            )
