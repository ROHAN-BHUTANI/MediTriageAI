"""Language identification API and heuristics for the MediTriageAI analysis framework."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


class BaseLanguageDetector(ABC):
    """Abstract base class for language detectors in the analysis framework.
    
    This interface allows future NLP-based language classification models to be 
    easily plugged in to replace the rule-based heuristics.
    """
    
    @abstractmethod
    def detect(self, text: str) -> str:
        """Detect the language of a given text.
        
        Args:
            text: The text sample to classify.
            
        Returns:
            A string identifier for the language, e.g., 'English', 'Hindi', 'Hinglish', 'Mixed', or 'Unknown'.
        """
        pass


class HeuristicLanguageDetector(BaseLanguageDetector):
    """Lightweight rule-based language detector using character sets and token heuristics."""

    def __init__(self) -> None:
        # Common Hinglish stopwords and phonetic subwords
        self.hinglish_keywords = {
            "hai", "he", "hy", "aur", "ko", "se", "me", "mein", "bhi", "kya", "dard", "darad", "ho", "raha", "rha",
            "bohot", "bahut", "bhut", "jyada", "zyada", "kal", "subah", "subha", "tabiyat", "tabiyyat", "shikayat",
            "ye", "yeh", "kaal", "ki", "ke", "ka", "hath", "sar", "pet", "pair", "bukhar", "khansi", "dawa", "bimar",
            "mera", "mujhe", "tum", "aap", "hu", "hoon", "tha", "thi", "the", "par", "pe", "kar", "karo",
            "rahi", "hoga", "hogi", "samajh", "bhai", "log", "kam", "kuch", "sab", "ghantey"
        }
        # Strong Hinglish markers that almost uniquely identify Hinglish text
        self.strong_hinglish_keywords = {
            "mera", "mujhe", "hai", "he", "dard", "darad", "ho", "rha", "raha", "bahut", "bohot", 
            "shikayat", "tabiyat", "tabiyyat"
        }

    def detect(self, text: str) -> str:
        """Categorize a clinical text as English, Hindi, Hinglish, Mixed, or Unknown."""
        if not isinstance(text, str) or not text.strip():
            return "Unknown"

        # Check for Devanagari characters (Hindi unicode range: U+0900 to U+097F)
        has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text)

        # Check for standard Latin characters
        has_latin = any(c.isalpha() and ord(c) < 128 for c in text)

        if has_devanagari and not has_latin:
            return "Hindi"
        elif has_devanagari and has_latin:
            return "Mixed"

        # Tokenize Latin characters
        words = [w.lower() for w in re.findall(r"[a-zA-Z]+", text)]
        if not words:
            return "Unknown"

        # Compute intersection with Hinglish keywords
        hinglish_hits = sum(1 for w in words if w in self.hinglish_keywords)
        has_strong_hit = any(w in self.strong_hinglish_keywords for w in words)

        # Classify as Hinglish if strong markers are present or if keyword density is high enough
        if has_strong_hit or hinglish_hits >= 2 or (len(words) > 0 and (hinglish_hits / len(words)) >= 0.08):
            return "Hinglish"

        # Default fallback for Latin text is English
        return "English"
