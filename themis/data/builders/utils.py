import re

from dataclasses import dataclass


@dataclass
class Token:
    text: str
    start: int
    end: int


@dataclass
class Affix:
    prefix: str
    infix_1: str
    infix_2: str
    suffix: str

    def to_template(self, mask_token: str = "<MASK>") -> dict:
        return {
            "template": f"{self.prefix}{mask_token}{self.suffix}",
            "infix_1": self.infix_1,
            "infix_2": self.infix_2,
        }


def tokenize(text: str) -> list[Token]:
    """
    Word-level tokenization with regex.
    Tokenizes the input text into a list of tokens.
    Also keeps track of start and end indices.
    """
    pattern = r"""
        \s+                       |  # whitespace
        \#\d+                     |  # hashtags or numeric IDs (e.g., #12345)
        \$[\d,]+(?:\.\d{1,2})?    |  # prices (e.g., $1,000.99 or $25.50)
        \w+(?:-\w+)*              |  # hyphens (e.g., state-of-the-art)
        \w+(?:'\w+)?              |  # contractions (e.g., isn't, don't) and possessives
        [^\w\s]                   |  # punctuation (single non-word non-space)
        \S                           # fallback
    """
    return [Token(m.group(), m.start(), m.end()) for m in re.finditer(pattern, text, re.VERBOSE)]


def find_diff_indices(tokens_1: list[Token], tokens_2: list[Token]) -> tuple[int, int, int]:
    """Finds the outermost bounds of all differing tokens."""
    min_len = min(len(tokens_1), len(tokens_2))

    # walk fowards - first difference
    start_diff = 0
    while start_diff < min_len and tokens_1[start_diff].text == tokens_2[start_diff].text:
        start_diff += 1

    # walk backwards - last difference
    end_diff_1, end_diff_2 = len(tokens_1), len(tokens_2)
    while end_diff_1 > start_diff and end_diff_2 > start_diff:
        if tokens_1[end_diff_1 - 1].text != tokens_2[end_diff_2 - 1].text:
            break
        end_diff_1 -= 1
        end_diff_2 -= 1

    return start_diff, end_diff_1, end_diff_2


def run_guards(sent_1: str, sent_2: str) -> Affix | None:
    """Common guards (identical sentences, empty sentences)."""
    if sent_1 == sent_2:
        return Affix(prefix=sent_1, infix_1="", infix_2="", suffix="")
    if not sent_1:
        return Affix(prefix="", infix_1="", infix_2=sent_2, suffix="")
    if not sent_2:
        return Affix(prefix="", infix_1=sent_1, infix_2="", suffix="")
    return None


def get_differences(sent_1: str, sent_2: str) -> Affix:
    """Finds differences between two sentences."""
    guard_diff = run_guards(sent_1, sent_2)
    if guard_diff:
        return guard_diff

    tokens_1, tokens_2 = map(tokenize, (sent_1, sent_2))
    start_diff, end_diff_1, end_diff_2 = find_diff_indices(tokens_1, tokens_2)

    prefix = sent_1[: tokens_1[start_diff].start] if start_diff < len(tokens_1) else sent_1
    infix_1 = (
        sent_1[tokens_1[start_diff].start : tokens_1[end_diff_1 - 1].end]
        if start_diff < len(tokens_1) and end_diff_1 > 0
        else ""
    )
    infix_2 = (
        sent_2[tokens_2[start_diff].start : tokens_2[end_diff_2 - 1].end]
        if start_diff < len(tokens_2) and end_diff_2 > 0
        else ""
    )
    suffix = sent_1[tokens_1[end_diff_1 - 1].end :] if 0 < end_diff_1 <= len(tokens_1) else ""

    return Affix(prefix=prefix, infix_1=infix_1, infix_2=infix_2, suffix=suffix)
