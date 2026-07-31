"""Augmentation Plugins – All 15 implementations.

Each plugin is a self-contained AugmentationPlugin subclass.
"""

from __future__ import annotations

import random
import re
from typing import Any

from reconstruction.augmentations import AugmentationPlugin

# ── Shared utilities ──────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\b\w+\b")

_MEDICAL_ABBREVS = {
    "patient": "pt", "shortness of breath": "sob", "blood pressure": "bp",
    "heart rate": "hr", "temperature": "temp", "emergency department": "ed",
    "diagnosis": "dx", "treatment": "tx", "history": "hx",
    "prescription": "rx", "symptoms": "sx", "examination": "exam",
    "follow up": "f/u", "complaint": "c/o", "with": "w/",
    "without": "w/o", "years old": "y/o", "left": "lt", "right": "rt",
}
_ABBREV_REVERSE = {v: k for k, v in _MEDICAL_ABBREVS.items()}

_SMS_MAP = {
    "you": "u", "are": "r", "your": "ur", "please": "pls",
    "because": "bcz", "before": "b4", "tomorrow": "tmrw",
    "today": "tdy", "problem": "prblm", "doctor": "dr",
    "medicine": "med", "hospital": "hosp", "feeling": "feling",
    "something": "smthng", "nothing": "nthng", "condition": "cond",
}

_HINDI_PHRASES = {
    "pain": "dard", "headache": "sar dard", "fever": "bukhar",
    "cough": "khansi", "cold": "zukam", "vomiting": "ulti",
    "dizziness": "chakkar", "weakness": "kamzori", "swelling": "sujan",
    "bleeding": "khoon", "stomach": "pet", "chest": "seena",
    "breathing": "saans", "injury": "chot", "wound": "ghav",
    "infection": "sankraman", "allergy": "allergy",
    "I have": "mujhe", "it hurts": "dard ho raha hai",
    "since yesterday": "kal se", "for two days": "do din se",
    "very bad": "bahut bura", "please help": "madad kijiye",
}

_KEYBOARD_NEIGHBORS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "rw",
    "f": "dg", "g": "fh", "h": "gj", "i": "uo", "j": "hk",
    "k": "jl", "l": "k", "m": "n", "n": "bm", "o": "ip",
    "p": "o", "q": "w", "r": "et", "s": "ad", "t": "ry",
    "u": "yi", "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}


# ── 1. English Lexical Rewrite ───────────────────────────────────────────

class EnglishLexicalRewrite(AugmentationPlugin):
    _SYNONYMS = {
        "pain": ["ache", "soreness", "discomfort", "hurt"],
        "severe": ["intense", "acute", "extreme", "sharp"],
        "mild": ["slight", "minor", "gentle", "moderate"],
        "injury": ["wound", "trauma", "damage", "hurt"],
        "swelling": ["inflammation", "puffiness", "edema", "bump"],
        "bleeding": ["hemorrhage", "blood loss", "hemorrhaging"],
        "vomiting": ["throwing up", "emesis", "nausea with vomiting"],
        "headache": ["head pain", "cephalalgia", "migraine"],
        "dizziness": ["vertigo", "lightheadedness", "unsteadiness"],
        "fever": ["elevated temperature", "pyrexia", "high temp"],
        "cough": ["coughing", "hacking cough", "persistent cough"],
        "fracture": ["break", "broken bone", "crack"],
    }

    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        result = text
        for word, syns in self._SYNONYMS.items():
            if re.search(rf"\b{word}\b", result, re.IGNORECASE):
                if rng.random() < 0.5:
                    replacement = rng.choice(syns)
                    result = re.sub(rf"\b{word}\b", replacement, result, count=1, flags=re.IGNORECASE)
        return result

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "EnglishLexicalRewrite", "version": "1.0", "description": "Synonym replacement for medical terms"}

    def supported_languages(self) -> list[str]:
        return ["en"]


# ── 2. Hindi Translation ─────────────────────────────────────────────────

class HindiTranslation(AugmentationPlugin):
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        result = text
        items = list(_HINDI_PHRASES.items())
        rng.shuffle(items)
        for eng, hindi in items:
            if eng.lower() in result.lower() and rng.random() < 0.6:
                result = re.sub(rf"\b{re.escape(eng)}\b", hindi, result, count=1, flags=re.IGNORECASE)
        return result

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "HindiTranslation", "version": "1.0", "description": "Partial English-to-Hindi translation"}

    def supported_languages(self) -> list[str]:
        return ["en", "hi-en"]


# ── 3. Roman Hindi Conversion ────────────────────────────────────────────

class RomanHindiConversion(AugmentationPlugin):
    _MAP = {
        "pain": "dard", "head": "sar", "stomach": "pet", "fever": "bukhar",
        "cold": "thand", "cough": "khansi", "doctor": "daktar",
        "medicine": "dawai", "hospital": "aspatal", "very": "bahut",
        "bad": "bura", "good": "accha", "day": "din", "night": "raat",
        "water": "paani", "food": "khana", "sleep": "neend",
    }

    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        result = text
        for eng, roman in self._MAP.items():
            if re.search(rf"\b{eng}\b", result, re.IGNORECASE) and rng.random() < 0.5:
                result = re.sub(rf"\b{eng}\b", roman, result, count=1, flags=re.IGNORECASE)
        return result

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "RomanHindiConversion", "version": "1.0", "description": "English words to Roman Hindi"}

    def supported_languages(self) -> list[str]:
        return ["en", "hi-en"]


# ── 4. Hinglish Conversion ───────────────────────────────────────────────

class HinglishConversion(AugmentationPlugin):
    _TEMPLATES = [
        "mujhe {symptom} ho raha hai", "mera {body} mein {symptom} hai",
        "{symptom} bahut zyada hai", "kal se {symptom} hai",
        "doctor sahab {symptom} ho gaya", "{symptom} se pareshan hoon",
    ]

    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        result = text
        for eng, hindi in _HINDI_PHRASES.items():
            if eng.lower() in result.lower() and rng.random() < 0.4:
                result = re.sub(rf"\b{re.escape(eng)}\b", hindi, result, count=1, flags=re.IGNORECASE)
        # Mix sentence structure
        if rng.random() < 0.3:
            result = result.rstrip(".") + " hai"
        return result

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "HinglishConversion", "version": "1.0", "description": "Code-mixed English-Hindi"}

    def supported_languages(self) -> list[str]:
        return ["en", "hi-en"]


# ── 5. Broken English ────────────────────────────────────────────────────

class BrokenEnglish(AugmentationPlugin):
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        words = text.split()
        result = []
        for w in words:
            if rng.random() < 0.15:
                continue  # Drop word
            if rng.random() < 0.1:
                w = w + " " + w  # Stutter
            result.append(w)
        out = " ".join(result)
        # Remove articles
        out = re.sub(r"\b(the|a|an)\b", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\s+", " ", out).strip()
        return out if out else text

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "BrokenEnglish", "version": "1.0", "description": "Non-native English simulation"}

    def supported_languages(self) -> list[str]:
        return ["en"]


# ── 6. Broken Hinglish ───────────────────────────────────────────────────

class BrokenHinglish(AugmentationPlugin):
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        # Apply Hindi conversion first
        result = text
        for eng, hindi in list(_HINDI_PHRASES.items())[:8]:
            if eng.lower() in result.lower() and rng.random() < 0.5:
                result = re.sub(rf"\b{re.escape(eng)}\b", hindi, result, count=1, flags=re.IGNORECASE)
        # Drop random words
        words = result.split()
        words = [w for w in words if rng.random() > 0.1]
        result = " ".join(words)
        return result if result else text

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "BrokenHinglish", "version": "1.0", "description": "Broken code-mixed Hindi-English"}

    def supported_languages(self) -> list[str]:
        return ["en", "hi-en"]


# ── 7. SMS Shorthand ─────────────────────────────────────────────────────

class SmsShorthand(AugmentationPlugin):
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        result = text.lower()
        for full, short in _SMS_MAP.items():
            if full in result and rng.random() < 0.6:
                result = result.replace(full, short, 1)
        # Remove vowels from some words
        words = result.split()
        out = []
        for w in words:
            if len(w) > 4 and rng.random() < 0.2:
                w = w[0] + re.sub(r"[aeiou]", "", w[1:])
            out.append(w)
        return " ".join(out)

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "SmsShorthand", "version": "1.0", "description": "SMS/texting abbreviations"}

    def supported_languages(self) -> list[str]:
        return ["en"]


# ── 8. ASR Corruption ────────────────────────────────────────────────────

class AsrCorruption(AugmentationPlugin):
    _HOMOPHONES = {
        "pain": ["pane", "payne"], "ache": ["ake", "aik"],
        "head": ["hed", "had"], "cold": ["code", "called"],
        "weak": ["week"], "break": ["brake"], "night": ["nite"],
        "weight": ["wait"], "right": ["write", "rite"],
        "hear": ["here"], "eye": ["I"], "heel": ["heal"],
        "patient": ["pashent", "payshent"],
    }

    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        result = text
        for word, replacements in self._HOMOPHONES.items():
            if re.search(rf"\b{word}\b", result, re.IGNORECASE) and rng.random() < 0.4:
                result = re.sub(rf"\b{word}\b", rng.choice(replacements), result, count=1, flags=re.IGNORECASE)
        # Insert filler words
        if rng.random() < 0.3:
            words = result.split()
            pos = rng.randint(0, max(0, len(words) - 1))
            filler = rng.choice(["um", "uh", "like", "you know"])
            words.insert(pos, filler)
            result = " ".join(words)
        return result

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "AsrCorruption", "version": "1.0", "description": "Speech-to-text error simulation"}

    def supported_languages(self) -> list[str]:
        return ["en"]


# ── 9. Keyboard Typo Corruption ──────────────────────────────────────────

class KeyboardTypo(AugmentationPlugin):
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        chars = list(text)
        n_typos = max(1, int(len(chars) * 0.05))
        for _ in range(n_typos):
            pos = rng.randint(0, len(chars) - 1)
            c = chars[pos].lower()
            if c in _KEYBOARD_NEIGHBORS:
                chars[pos] = rng.choice(list(_KEYBOARD_NEIGHBORS[c]))
        return "".join(chars)

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "KeyboardTypo", "version": "1.0", "description": "Keyboard proximity typo injection"}

    def supported_languages(self) -> list[str]:
        return ["en", "hi-en"]


# ── 10. Medical Abbreviation Expansion ───────────────────────────────────

class MedicalAbbreviationExpansion(AugmentationPlugin):
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        result = text
        for abbrev, full in _ABBREV_REVERSE.items():
            if re.search(rf"\b{re.escape(abbrev)}\b", result, re.IGNORECASE) and rng.random() < 0.6:
                result = re.sub(rf"\b{re.escape(abbrev)}\b", full, result, count=1, flags=re.IGNORECASE)
        return result

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "MedicalAbbreviationExpansion", "version": "1.0", "description": "Expand medical abbreviations"}

    def supported_languages(self) -> list[str]:
        return ["en"]


# ── 11. Medical Abbreviation Contraction ─────────────────────────────────

class MedicalAbbreviationContraction(AugmentationPlugin):
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        result = text
        for full, abbrev in _MEDICAL_ABBREVS.items():
            if full.lower() in result.lower() and rng.random() < 0.6:
                result = re.sub(rf"\b{re.escape(full)}\b", abbrev, result, count=1, flags=re.IGNORECASE)
        return result

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "MedicalAbbreviationContraction", "version": "1.0", "description": "Contract to medical abbreviations"}

    def supported_languages(self) -> list[str]:
        return ["en"]


# ── 12. Punctuation Removal ──────────────────────────────────────────────

class PunctuationRemoval(AugmentationPlugin):
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        if rng.random() < 0.5:
            return re.sub(r"[^\w\s]", "", text)
        else:
            return re.sub(r"[.,;:!?]", "", text)

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "PunctuationRemoval", "version": "1.0", "description": "Remove punctuation"}

    def supported_languages(self) -> list[str]:
        return ["en", "hi-en"]


# ── 13. Capitalization Variation ─────────────────────────────────────────

class CapitalizationVariation(AugmentationPlugin):
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        choice = rng.randint(0, 3)
        if choice == 0:
            return text.lower()
        elif choice == 1:
            return text.upper()
        elif choice == 2:
            return text.title()
        else:
            return "".join(c.upper() if rng.random() < 0.3 else c.lower() for c in text)

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "CapitalizationVariation", "version": "1.0", "description": "Random case variation"}

    def supported_languages(self) -> list[str]:
        return ["en", "hi-en"]


# ── 14. Symptom Order Permutation ────────────────────────────────────────

class SymptomOrderPermutation(AugmentationPlugin):
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        separators = [",", " and ", ";", " with "]
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                rng.shuffle(parts)
                return sep.join(parts)
        # Fallback: shuffle sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            rng.shuffle(sentences)
            return " ".join(sentences)
        return text

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "SymptomOrderPermutation", "version": "1.0", "description": "Reorder listed symptoms"}

    def supported_languages(self) -> list[str]:
        return ["en", "hi-en"]


# ── 15. Clinically Equivalent Rewrite ────────────────────────────────────

class ClinicallyEquivalentRewrite(AugmentationPlugin):
    _REWRITES = {
        "my head hurts": "I have a headache",
        "i have a headache": "my head is pounding",
        "stomach pain": "abdominal discomfort",
        "can't breathe": "difficulty breathing",
        "difficulty breathing": "shortness of breath",
        "threw up": "vomited",
        "feeling dizzy": "experiencing vertigo",
        "cut my": "lacerated my",
        "broken": "fractured",
        "really bad": "severe",
        "a lot of pain": "significant pain",
        "hurts a lot": "experiencing severe pain",
    }

    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        rng = random.Random(seed)
        result = text
        items = list(self._REWRITES.items())
        rng.shuffle(items)
        for original, rewrite in items:
            if original.lower() in result.lower() and rng.random() < 0.7:
                result = re.sub(re.escape(original), rewrite, result, count=1, flags=re.IGNORECASE)
        return result

    def plugin_metadata(self) -> dict[str, Any]:
        return {"name": "ClinicallyEquivalentRewrite", "version": "1.0", "description": "Semantically equivalent clinical rephrasing"}

    def supported_languages(self) -> list[str]:
        return ["en"]


# ── Plugin Registry ──────────────────────────────────────────────────────

ALL_PLUGINS: list[type[AugmentationPlugin]] = [
    EnglishLexicalRewrite,
    HindiTranslation,
    RomanHindiConversion,
    HinglishConversion,
    BrokenEnglish,
    BrokenHinglish,
    SmsShorthand,
    AsrCorruption,
    KeyboardTypo,
    MedicalAbbreviationExpansion,
    MedicalAbbreviationContraction,
    PunctuationRemoval,
    CapitalizationVariation,
    SymptomOrderPermutation,
    ClinicallyEquivalentRewrite,
]


def get_all_plugins() -> list[AugmentationPlugin]:
    """Instantiate all registered augmentation plugins."""
    return [cls() for cls in ALL_PLUGINS]
