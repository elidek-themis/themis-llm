import ast
import re
from typing import Any, Dict, Generator, List, Tuple, Union

import datasets
import pandas as pd

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


class PersonaConfig(datasets.BuilderConfig):
    def __init__(
        self,
        template: str | List[Dict[str, str]],
        sub: Dict[str, str] | None = None,
        choices: List[str] = [],
        columns: List[str] = [],
        **kwargs,
    ):

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

    def __init__(self, **config) -> None:
        super().__init__(**config)
        template = self.config.template

        if isinstance(template, str):
            self.is_chat = False
        elif isinstance(template, list):
            self.is_chat = True
        else:
            raise ValueError("Template should be a string or a list of dictionaries")

        self.choices = self.config.choices
        self.columns = self.config.columns

        self.template = str(template)
        assert "{persona}" in self.template, "template should include {persona} in brackets"

        self.sub = self.config.sub

    def _info(self):
        chat = [datasets.Features({"role": datasets.Value("string"), "content": datasets.Value("string")})]

        features = datasets.Features(
            {
                "key": datasets.Value("string"),
                "input": datasets.Value("string") if isinstance(self.config.template, str) else chat,
                "choices": datasets.Sequence(datasets.Value("string")),
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
            return [datasets.SplitGenerator(name="test", gen_kwargs={"split": "residency"})]
        elif self.config.name == "demographic":
            return [
                datasets.SplitGenerator(
                    name="test", gen_kwargs={"split": "demographic", "path": "themis/data/persona/demographics.csv"}
                )
            ]

    def _generate_examples(self, split: str, path=None):  # -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        print(self.config.name)
        if split == "residency":
            return self._generate_residency_examples()
        elif split == "demographic":
            return self._generate_demographic_examples(path)

    def _format(self, persona: str) -> Union[str, List[dict]]:
        if self.is_chat:
            return ast.literal_eval(re.sub(string=self.template, pattern="{persona}", repl=persona))
        else:
            return self.template.format(persona=persona)

    def _generate_residency_examples(self):  # -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        for i, state in enumerate(_STATES):
            persona = self._format(persona=state)
            yield i, {"key": state, "input": persona, "choices": self.choices}

        persona = re.sub(string=self.template, pattern=self.sub["pattern"], repl=self.sub["repl"])
        persona = ast.literal_eval(persona) if self.is_chat else persona
        yield 51, {"key": "U.S.", "input": persona, "choices": self.choices}

    def _generate_demographic_examples(self, path):  # -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        demographics = pd.read_csv(path, sep=",", quotechar='"', skipinitialspace=True)
        for i, row in demographics.iterrows():
            key = f"{row['demographic']} - {row['group']}"
            persona = self.template(persona=row["persona"])
            yield i, {"key": key, "persona": persona, "choices": self.config.choices}
