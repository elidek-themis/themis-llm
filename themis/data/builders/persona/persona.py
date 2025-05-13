import re

from typing import Any

import pandas as pd
import datasets

_DESCRIPTION = "U.S. Elections Prompts for Political Polling"
_HOMEPAGE = ""
_LICENSE = ""
_CITATION = ""
_URL = ""

_STATES = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]

_DEMOGRAPHICS_PATH = "themis/data/persona/demographics.csv"


class PersonaConfig(datasets.BuilderConfig):
    def __init__(
        self,
        template: str,
        sub: dict[str, str] | None = None,
        choices: dict[str, list] | None = None,
        columns: list[str] | None = None,
        **kwargs: Any,
    ):
        if choices is None:
            choices = {}
        if columns is None:
            columns = []

        super().__init__(**kwargs)

        self.template = template
        self.sub = sub
        self.choices = choices
        self.columns = columns


class Persona(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        PersonaConfig(
            name="residency",
            template="{persona}",
            sub={"pattern": "{persona}", "repl": "the U.S."},
            version=VERSION,
            description="residency-persona",
        ),
        PersonaConfig(
            name="demographic",
            template="{persona}",
            version=VERSION,
            description="demographic-persona",
        ),
    ]

    DEFAULT_CONFIG_NAME = "residency"

    def __init__(self, **config: PersonaConfig):
        super().__init__(**config)
        self.template = self.config.template
        assert "{persona}" in self.template, "template should include {persona} in brackets"

        self.choices = self.config.choices
        self.columns = self.config.columns
        self.sub = self.config.sub

    def _info(self):
        choices = datasets.Features(
            {
                "pro": datasets.Sequence(datasets.Value("string")),
                "contra": datasets.Sequence(datasets.Value("string")),
            }
        )

        features = datasets.Features(
            {
                "key": datasets.Value("string"),
                "template": datasets.Value("string"),
                "choices": choices,
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
        if self.config.name == "residency":
            return [datasets.SplitGenerator(name="residency", gen_kwargs={"split": "residency"})]
        elif self.config.name == "demographic":
            return [datasets.SplitGenerator(name="demographic", gen_kwargs={"split": "demographic"})]

        raise ValueError(f"Unknown config name: {self.config.name}")

    def _generate_examples(self, split: str):
        if split == "residency":
            return self._generate_residency_examples()
        elif split == "demographic":
            return self._generate_demographic_examples()

        raise NotImplementedError(f"Unknown split: {split}")

    def _generate_residency_examples(self):
        for i, state in enumerate(_STATES):
            persona = self.template.format(persona=state)
            yield (
                i,
                {
                    "key": state,
                    "template": persona,
                    "choices": self.choices,
                },
            )

        # U.S citizenship
        persona = re.sub(string=self.template, pattern=self.sub["pattern"], repl=self.sub["repl"])
        yield 51, {"key": "U.S.", "template": persona, "choices": self.choices}

    def _generate_demographic_examples(self):
        demographics = pd.read_csv(_DEMOGRAPHICS_PATH, sep=",", quotechar='"', skipinitialspace=True)
        for i, row in demographics.iterrows():
            key = f"{row['demographic']} - {row['group']}"
            persona = self.template.format(persona=row["persona"])
            yield i, {"key": key, "template": persona, "choices": self.config.choices}
