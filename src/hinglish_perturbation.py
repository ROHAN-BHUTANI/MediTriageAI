"""Deterministic Hinglish perturbation helpers for MediTriageAI."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhoneticVariant:
    pattern: re.Pattern[str]
    alternatives: tuple[str, ...]
    description: str
    canonical: str


def _compile_word(word: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


_VARIANT_TABLE: tuple[PhoneticVariant, ...] = (
    PhoneticVariant(_compile_word("hai"), ("hain", "he", "hy"), "is/am/are (copula)", "hai"),
    PhoneticVariant(_compile_word("nahi"), ("nahin", "nai", "nhi"), "no/not", "nahi"),
    PhoneticVariant(_compile_word("nahin"), ("nahi", "nai", "nhi"), "no/not", "nahin"),
    PhoneticVariant(_compile_word("kal"), ("kaal",), "yesterday/tomorrow", "kal"),
    PhoneticVariant(_compile_word("kya"), ("kia", "kyaa"), "what", "kya"),
    PhoneticVariant(_compile_word("mera"), ("meraa", "mera"), "my (masc.)", "mera"),
    PhoneticVariant(_compile_word("meri"), ("meree", "meri"), "my (fem.)", "meri"),
    PhoneticVariant(_compile_word("aap"), ("ap", "aaap"), "you (formal)", "aap"),
    PhoneticVariant(_compile_word("aapka"), ("apka", "aapkaa"), "your (formal)", "aapka"),
    PhoneticVariant(_compile_word("bahut"), ("bohot", "bahot", "bhut"), "very/a lot", "bahut"),
    PhoneticVariant(_compile_word("bohot"), ("bahut", "bahot", "bhut"), "very/a lot", "bohot"),
    PhoneticVariant(_compile_word("dard"), ("dardh", "darad"), "pain", "dard"),
    PhoneticVariant(_compile_word("tabiyat"), ("tabiyyat", "tabiyat"), "health/condition", "tabiyat"),
    PhoneticVariant(_compile_word("theek"), ("thik", "theeq", "tik"), "fine/okay", "theek"),
    PhoneticVariant(_compile_word("zyada"), ("jyada", "jiyada", "ziyada"), "more (z/j borrowed-sound variant)", "zyada"),
    PhoneticVariant(_compile_word("zindagi"), ("jindagi", "zindgi"), "life (z/j borrowed-sound variant)", "zindagi"),
    PhoneticVariant(_compile_word("ho"), ("hoo",), "be/happen", "ho"),
    PhoneticVariant(_compile_word("raha"), ("rha", "rehaa"), "continuous-aspect particle (masc.)", "raha"),
    PhoneticVariant(_compile_word("rahi"), ("rhi", "rehee"), "continuous-aspect particle (fem.)", "rahi"),
    PhoneticVariant(_compile_word("samay"), ("samaya", "samai"), "time", "samay"),
    PhoneticVariant(_compile_word("subah"), ("subha", "subaha"), "morning", "subah"),
    PhoneticVariant(_compile_word("raat"), ("rat", "raaat"), "night", "raat"),
    PhoneticVariant(_compile_word("doctor"), ("daktar", "dactor"), "doctor (borrowed-word respelling)", "doctor"),
    PhoneticVariant(_compile_word("hospital"), ("aspataal", "haspatal"), "hospital (borrowed-word respelling)", "hospital"),
    PhoneticVariant(_compile_word("medicine"), ("medecine", "medisin"), "medicine (borrowed-word respelling)", "medicine"),
)

_FINAL_H_DROP_WORDS: tuple[str, ...] = ("yeh", "voh", "kuch", "sab", "thoda")
_FINAL_H_DROP_REPLACEMENTS: dict[str, str] = {"yeh": "ye", "voh": "vo"}


@dataclass
class PerturbationResult:
    original: str
    perturbed: str
    substitutions_applied: list[tuple[str, str, str]] = field(default_factory=list)
    seed: int = 0


def _match_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def perturb_text(text: str, seed: int, *, substitution_rate: float = 0.5) -> PerturbationResult:
    if not (0.0 <= substitution_rate <= 1.0):
        raise ValueError(f"substitution_rate must be in [0,1], got {substitution_rate}")
    rng = random.Random(seed)
    substitutions: list[tuple[str, str, str]] = []

    def _make_replacer(variant: PhoneticVariant):
        def _replace(match: re.Match[str]) -> str:
            if rng.random() > substitution_rate:
                return match.group(0)
            choice = _match_case(match.group(0), rng.choice(variant.alternatives))
            substitutions.append((match.group(0), choice, variant.description))
            return choice

        return _replace

    perturbed = text
    for variant in _VARIANT_TABLE:
        perturbed = variant.pattern.sub(_make_replacer(variant), perturbed)

    for word in _FINAL_H_DROP_WORDS:
        pattern = _compile_word(word)
        replacement = _FINAL_H_DROP_REPLACEMENTS.get(word, word[:-1])

        def _h_drop_replace(match: re.Match[str], _repl=replacement) -> str:
            if rng.random() > substitution_rate:
                return match.group(0)
            cased_repl = _match_case(match.group(0), _repl)
            substitutions.append((match.group(0), cased_repl, "word-final h-dropping"))
            return cased_repl

        perturbed = pattern.sub(_h_drop_replace, perturbed)

    return PerturbationResult(original=text, perturbed=perturbed, substitutions_applied=substitutions, seed=seed)
