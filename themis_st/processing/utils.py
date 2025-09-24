import re

from jinja2 import Environment, StrictUndefined


# e.g. {{ "foo123bar" | regex_replace("[0-9]+", "XYZ") }}
def regex_replace(string, pattern, repl, count: int = 0):
    """Implements the `re.sub` function as a custom Jinja filter."""
    return re.sub(pattern, repl, string, count=count)


env = Environment(undefined=StrictUndefined)
env.filters["regex_replace"] = regex_replace


def apply_template(template: str, doc: dict) -> str:
    rtemplate = env.from_string(template)
    return rtemplate.render(**doc)


def format_jinja(template: str, indent_width: int = 4) -> str:
    token_re = re.compile(r"({{.*?}}|{%-?.*?-%}|{%.*?%})", re.DOTALL)
    tokens = token_re.findall(template)
    indent = 0
    result = []

    for token in tokens:
        stripped = token.strip()

        # dedent before writing if it's an end tag
        if (
            re.match(r"{%[-\s]*end\w+", stripped)
            or re.match(r"{%[-\s]*else", stripped)
            or re.match(r"{%[-\s]*elif", stripped)
        ):
            indent -= 1

        result.append(" " * (indent * indent_width) + stripped)

        # re-indent after if it's an opening tag (not else/elif/end)
        if re.match(r"{%[-\s]*(for|if|block|macro|filter|with)\b", stripped):
            indent += 1
        elif re.match(r"{%[-\s]*(else|elif)\b", stripped):
            indent += 1

    return "\n".join(result)
