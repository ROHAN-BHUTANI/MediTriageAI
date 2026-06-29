"""
hinglish_perturbation.py
--------------------------
Deterministic phonetic perturbation engine for injecting realistic Hinglish
(romanized Hindi-English code-mixed) spelling noise into clinical text, so the
trained classifier sees text that resembles real patient self-report language
rather than the artificially clean prose naive unconstrained LLM generation
tends to produce.

Grounding: real Hindi-English code-mixed text has well-documented, recurring
spelling-variation classes because there is no single standard for romanizing
Hindi (Bhargava et al., "Automatic Normalization of Word Variations in
Code-Mixed Social Media Text", arXiv:1804.00804). The two classes we encode
here are taken directly from that literature, not invented:

  1. Long-vowel transliteration variation: a long vowel sound may be written
     by doubling the vowel ("khaaya"), by capitalizing it ("khAya" -- we do
     NOT use case variation since clinical/chat text is usually all-lowercase
     in practice, so we use vowel-doubling/dropping instead), or by omitting
     the length marker entirely ("khaya"). E.g. साल -> saal / sal;
     मेरा -> meraa / mera; आपका -> aapka / apka.
  2. Borrowed/foreign-sound variation: sounds without an exact Hindi
     phonemic counterpart get inconsistent renderings, e.g. ज़ (z-sound) vs
     ज (j-sound): izzat/ijjat, azad/ajad, zindabad/jindabad.

We also encode the extremely common "h-dropping/adding" and consonant-cluster
simplification patterns documented in the "ye jaruri hai" -> 64-spellings
example (Bollyrics, arXiv:2007.12916): y/ye/yeh alternation, j/z alternation,
short/long vowel alternation, and word-final h-dropping.

This is a DETERMINISTIC engine: given the same (text, seed) pair it always
produces the same output. This is required for reproducible dataset builds
and for the leakage-safe split (a perturbed variant of a training-set sample
must never be able to land in the validation/test split -- see
src/leakage_safe_split.py). Determinism is achieved via a local
random.Random(seed) instance seeded per-call, never the global random module.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# --- Phonetic equivalence classes -----------------------------------------
# Each entry: a regex matching one "canonical-ish" spelling, and a list of
# real, documented alternative spellings it may be replaced with. All
# matching is case-insensitive and word-boundary-anchored so we only touch
# whole words, never substrings inside unrelated words.

@dataclass(frozen=True)
class PhoneticVariant:
    pattern: re.Pattern[str]
    alternatives: tuple[str, ...]
    description: str


def _compile_word(word: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


# Common Hinglish function words and clinical-conversation words with
# well-attested spelling variants. This list is intentionally a *curated,
# documented* set (not exhaustive of all Hindi vocabulary) -- it covers the
# function words and symptom-adjacent words most likely to appear in a
# patient's own free-text description of how they feel, which is the
# register this perturbation engine targets (NOT formal medical terminology,
# which stays in English in real Hinglish clinical self-reports).
_VARIANT_TABLE: tuple[PhoneticVariant, ...] = (
    PhoneticVariant(_compile_word("hai"), ("hain", "he", "hy"), "is/am/are (copula)"),
    PhoneticVariant(_compile_word("nahi"), ("nahin", "nai", "nhi"), "no/not"),
    PhoneticVariant(_compile_word("nahin"), ("nahi", "nai", "nhi"), "no/not"),
    PhoneticVariant(_compile_word("kal"), ("kaal",), "yesterday/tomorrow"),
    PhoneticVariant(_compile_word("kya"), ("kia", "kyaa"), "what"),
    PhoneticVariant(_compile_word("mera"), ("meraa", "mera"), "my (masc.)"),
    PhoneticVariant(_compile_word("meri"), ("meree", "meri"), "my (fem.)"),
    PhoneticVariant(_compile_word("aap"), ("ap", "aaap"), "you (formal)"),
    PhoneticVariant(_compile_word("aapka"), ("apka", "aapkaa"), "your (formal)"),
    PhoneticVariant(_compile_word("bahut"), ("bohot", "bahot", "bhut"), "very/a lot"),
    PhoneticVariant(_compile_word("bohot"), ("bahut", "bahot", "bhut"), "very/a lot"),
    PhoneticVariant(_compile_word("dard"), ("dardh", "darad"), "pain"),
    PhoneticVariant(_compile_word("tabiyat"), ("tabiyyat", "tabiyat"), "health/condition"),
    PhoneticVariant(_compile_word("theek"), ("thik", "theeq", "tik"), "fine/okay"),
    PhoneticVariant(_compile_word("zyada"), ("jyada", "jiyada", "ziyada"), "more (z/j borrowed-sound variant)"),
    PhoneticVariant(_compile_word("zindagi"), ("jindagi", "zindgi"), "life (z/j borrowed-sound variant)"),
    PhoneticVariant(_compile_word("ho"), ("hoo",), "be/happen"),
    PhoneticVariant(_compile_word("raha"), ("rha", "rehaa"), "continuous-aspect particle (masc.)"),
    PhoneticVariant(_compile_word("rahi"), ("rhi", "rehee"), "continuous-aspect particle (fem.)"),
    PhoneticVariant(_compile_word("samay"), ("samaya", "samai"), "time"),
    PhoneticVariant(_compile_word("subah"), ("subha", "subaha"), "morning"),
    PhoneticVariant(_compile_word("raat"), ("rat", "raaat"), "night"),
    PhoneticVariant(_compile_word("doctor"), ("daktar", "dactor"), "doctor (borrowed-word respelling)"),
    PhoneticVariant(_compile_word("hospital"), ("aspataal", "haspatal"), "hospital (borrowed-word respelling)"),
    PhoneticVariant(_compile_word("medicine"), ("medecine", "medisin"), "medicine (borrowed-word respelling)"),
)

# Word-final "h" dropping: a documented, very common pattern (e.g. "yeh" -> "ye").
_FINAL_H_DROP_WORDS: tuple[str, ...] = ("yeh", "voh", "kuch", "sab", "thoda")
_FINAL_H_DROP_REPLACEMENTS: dict[str, str] = {
    "yeh": "ye",
    "voh": "vo",
}


@dataclass
class PerturbationResult:
    original: str
    perturbed: str
    substitutions_applied: list[tuple[str, str, str]] = field(default_factory=list)
    # (original_word, replacement_word, description) per substitution
    seed: int = 0


def _match_case(original: str, replacement: str) -> str:
    """
    Apply the capitalization pattern of `original` to `replacement`, so that
    e.g. "Mera" -> "mera" (a real alternative spelling) renders as "Mera"
    when the source word was capitalized, rather than silently lowercasing
    sentence-initial words during substitution.
    """
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def perturb_text(
    text: str,
    seed: int,
    *,
    substitution_rate: float = 0.5,
) -> PerturbationResult:
    """
    Apply deterministic phonetic perturbation to `text`.

    Parameters
    ----------
    text:
        Input text (expected to already contain Hinglish function words from
        the variant table -- this engine perturbs existing Hinglish spelling,
        it does not code-mix pure English text on its own).
    seed:
        Integer seed. The SAME (text, seed) pair always yields the SAME
        output -- this is load-bearing for leakage-safe dataset construction
        (a perturbed variant must be deterministically traceable back to its
        source row's tracking ID; see src/leakage_safe_split.py).
    substitution_rate:
        Probability, per eligible word match, that a variant substitution is
        applied. 0.0 = no perturbation (returns text unchanged). 1.0 = every
        eligible match is perturbed. Default 0.5 models realistic noise
        (i.e. not every instance of "hai" in real text gets misspelled the
        same way, or at all).

    Returns
    -------
    PerturbationResult
        Includes the full list of substitutions applied, for auditability and
        for unit testing.
    """
    if not (0.0 <= substitution_rate <= 1.0):
        raise ValueError(f"substitution_rate must be in [0,1], got {substitution_rate}")

    rng = random.Random(seed)  # local instance -- NEVER touch global random state
    substitutions: list[tuple[str, str, str]] = []

    def _make_replacer(variant: PhoneticVariant):
        def _replace(match: re.Match[str]) -> str:
            if rng.random() > substitution_rate:
                return match.group(0)
            choice = rng.choice(variant.alternatives)
            choice = _match_case(match.group(0), choice)
            substitutions.append((match.group(0), choice, variant.description))
            return choice

        return _replace

    perturbed = text
    for variant in _VARIANT_TABLE:
        perturbed = variant.pattern.sub(_make_replacer(variant), perturbed)

    # Final-h dropping pass (separate because it's a word-level full
    # replacement keyed by lookup table, not an alternatives list).
    for word in _FINAL_H_DROP_WORDS:
        pattern = _compile_word(word)
        replacement = _FINAL_H_DROP_REPLACEMENTS.get(word, word[:-1])  # drop trailing h

        def _h_drop_replace(match: re.Match[str], _word=word, _repl=replacement) -> str:
            if rng.random() > substitution_rate:
                return match.group(0)
            cased_repl = _match_case(match.group(0), _repl)
            substitutions.append((match.group(0), cased_repl, "word-final h-dropping"))
            return cased_repl

        perturbed = pattern.sub(_h_drop_replace, perturbed)

    return PerturbationResult(
        original=text,
        perturbed=perturbed,
        substitutions_applied=substitutions,
        seed=seed,
    )


def is_deterministic(text: str, seed: int, n_repeats: int = 5) -> bool:
    """
    Sanity-check helper: re-running perturb_text with the same (text, seed)
    must always produce byte-identical output. Used in tests and as a guard
    against accidental use of global random state.
    """
    first = perturb_text(text, seed).perturbed
    return all(perturb_text(text, seed).perturbed == first for _ in range(n_repeats))


if __name__ == "__main__":
    samples = [
        "Mera bahut dard ho raha hai, kal subah se theek nahi hun.",
        "Aapka tabiyat kaisi hai? Hospital jana padega kya?",
        "Yeh dard bohot zyada hai, raat ko sone nahi de raha.",
    ]

    print("Determinism check:")
    for s in samples:
        for seed in (1, 42, 999):
            ok = is_deterministic(s, seed)
            print(f"  seed={seed:4d} deterministic={ok}  text={s[:40]!r}")

    print("\nExample perturbations (seed=42):")
    for s in samples:
        result = perturb_text(s, seed=42, substitution_rate=0.6)
        print(f"  ORIGINAL : {result.original}")
        print(f"  PERTURBED: {result.perturbed}")
        print(f"  SUBS     : {result.substitutions_applied}")
        print()

    print("Different seeds give different (but each internally deterministic) outputs:")
    s = samples[0]
    for seed in (1, 2, 3):
        r = perturb_text(s, seed=seed, substitution_rate=0.6)
        print(f"  seed={seed}: {r.perturbed}")
