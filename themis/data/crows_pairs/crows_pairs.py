# coding=utf-8
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

import datasets
import pandas as pd


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

_DESCRIPTION = """\
CrowS-Pairs, a challenge dataset for measuring the degree to which U.S. stereotypical biases present in the masked language models (MLMs).
"""


_URLS = [ "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv",
         "https://raw.githubusercontent.com/nyu-mll/crows-pairs/refs/heads/master/data/prompts.csv"]

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


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class CrowsPairsPrompts(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="test", version=VERSION, description="CrowS-Pairs")
        # datasets.BuilderConfig(name="age", version=VERSION, description="CrowS-Pairs age"),
    ]

    DEFAULT_CONFIG_NAME = "test"

    def _info(self):
        features = datasets.Features(
            {
                "sent_more": datasets.Value("string"),
                "sent_less": datasets.Value("string"),
                "bias_type": datasets.Value("string"),
                "prompt": datasets.Value("string")                
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        data_files = dl_manager.download_and_extract(_URLS)

        return [
            # datasets.SplitGenerator(
            #     name=datasets.Split.TRAIN,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepaths": data_files,
            #         "split": "train",
            #     },
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": data_files,
                    "split": "test"
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "dev.jsonl"),
            #         "split": "dev",
            #     },
            # ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepaths, split):
        cols = ["sent_more", "sent_less", "stereo_antistereo", "bias_type"]

        data_path, prompts_path = filepaths

        df = pd.read_csv(data_path, usecols=cols)
        prompts = pd.read_csv(prompts_path)["prompt"]
        df = pd.concat([df, prompts], axis=1)

        # df = df[df.bias_type==split].drop("bias_type", axis=1)
        for key, row in enumerate(df.to_dict(orient="records")):
            yield key, row