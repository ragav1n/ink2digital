"""
German language post-processing for OCR output.

Handles:
- Umlaut correction (ae->ä, oe->ö, ue->ü, ss->ß)
- Common OCR error correction for German text
- Spell-checking using pyspellchecker
- German-specific character normalization
"""

from __future__ import annotations

import re
from typing import List, Optional, Dict
from pathlib import Path

# ---------------------------------------------------------------------------
# Umlaut and German character mappings
# ---------------------------------------------------------------------------

# Common OCR substitutions for German umlauts
UMLAUT_PATTERNS = [
    # Uppercase umlauts
    (re.compile(r'\bAe\b'), 'Ä'),
    (re.compile(r'\bOe\b'), 'Ö'),
    (re.compile(r'\bUe\b'), 'Ü'),
    # Lowercase umlauts (context-dependent, conservative)
    (re.compile(r'\bae\b'), 'ä'),
    (re.compile(r'\boe\b'), 'ö'),
    (re.compile(r'\bue\b'), 'ü'),
    # ß substitution
    (re.compile(r'ss(?=[aeiouäöü])', re.IGNORECASE), 'ß'),
]

# Common OCR confusions for German text
OCR_CORRECTIONS: Dict[str, str] = {
    # Digit/letter confusions
    '0': 'O',  # Only in word context — handled by spell checker
    'l': '1',  # Contextual
    # Common substitutions
    'rn': 'm',
    'vv': 'w',
    'ii': 'n',
    # German-specific
    'oe': 'ö',
    'ae': 'ä',
    'ue': 'ü',
    'Oe': 'Ö',
    'Ae': 'Ä',
    'Ue': 'Ü',
}

# German-specific word-level corrections for math/lecture content
DOMAIN_CORRECTIONS: Dict[str, str] = {
    'sei': 'sei',
    'fuer': 'für',
    'Fuer': 'Für',
    'ueber': 'über',
    'Ueber': 'Über',
    'unter': 'unter',
    'waehle': 'wähle',
    'Waehle': 'Wähle',
    'gilt': 'gilt',
    'Folge': 'Folge',
    'Reihe': 'Reihe',
    'Grenzwert': 'Grenzwert',
    'Ableitung': 'Ableitung',
    'Integral': 'Integral',
    'Funktion': 'Funktion',
    'Beweis': 'Beweis',
    'Definition': 'Definition',
    'Satz': 'Satz',
    'Lemma': 'Lemma',
    'Korollar': 'Korollar',
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to standard German representations."""
    import unicodedata
    # Normalize to NFC (composed form) for consistent umlaut representation
    return unicodedata.normalize('NFC', text)


def fix_umlaut_substitutions(text: str) -> str:
    """
    Fix common umlaut substitutions in OCR output.
    Conservative: only fixes obvious standalone cases.
    """
    for pattern, replacement in UMLAUT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def fix_domain_words(text: str) -> str:
    """Fix common domain-specific German words in math/CS lectures."""
    words = text.split()
    fixed = []
    for word in words:
        # Strip punctuation for lookup
        stripped = word.strip('.,;:!?()[]{}"\'-')
        if stripped in DOMAIN_CORRECTIONS:
            word = word.replace(stripped, DOMAIN_CORRECTIONS[stripped])
        fixed.append(word)
    return ' '.join(fixed)


def correct_german_ocr(text: str, use_spellcheck: bool = False) -> str:
    """
    Full pipeline for German OCR post-processing.

    Args:
        text: Raw OCR output string.
        use_spellcheck: Whether to apply spell-checker (requires pyspellchecker).

    Returns:
        Corrected text string.
    """
    if not text or not text.strip():
        return text

    # Step 1: Unicode normalization
    text = normalize_unicode(text)

    # Step 2: Fix umlaut substitutions
    text = fix_umlaut_substitutions(text)

    # Step 3: Fix domain-specific words
    text = fix_domain_words(text)

    # Step 4: Optional spell-checking
    if use_spellcheck:
        text = spellcheck_german(text)

    # Step 5: Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def spellcheck_german(text: str) -> str:
    """
    Apply German spell-checking using pyspellchecker.
    Only corrects high-confidence mistakes.
    """
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker(language='de')
    except (ImportError, Exception):
        return text  # Graceful fallback if not installed

    words = text.split()
    corrected = []
    for word in words:
        # Only spell-check alphabetic words (skip numbers, math)
        clean = re.sub(r'[^a-zA-ZäöüÄÖÜß]', '', word)
        if len(clean) < 3 or not clean.isalpha():
            corrected.append(word)
            continue

        candidates = spell.candidates(clean)
        if candidates:
            best = spell.correction(clean)
            if best and best != clean:
                word = word.replace(clean, best)
        corrected.append(word)
    return ' '.join(corrected)


def batch_correct(texts: List[str], use_spellcheck: bool = False) -> List[str]:
    """Apply German OCR correction to a batch of strings."""
    return [correct_german_ocr(t, use_spellcheck=use_spellcheck) for t in texts]


# ---------------------------------------------------------------------------
# Math-aware post-processing
# ---------------------------------------------------------------------------

def separate_text_and_math(text: str) -> List[dict]:
    """
    Separate inline math from plain text in OCR output.
    Identifies patterns like $...$, \\(...\\), numbers, operators.

    Returns:
        List of {'type': 'text'|'math', 'content': str}
    """
    # Simple pattern: detect math-like segments
    math_pattern = re.compile(
        r'(\$[^$]+\$|\\[(][^)]+\\[)]|[0-9]+[\+\-\*/\^=][0-9]+)'
    )
    segments = []
    last = 0
    for m in math_pattern.finditer(text):
        if m.start() > last:
            segments.append({'type': 'text', 'content': text[last:m.start()]})
        segments.append({'type': 'math', 'content': m.group()})
        last = m.end()
    if last < len(text):
        segments.append({'type': 'text', 'content': text[last:]})
    return segments if segments else [{'type': 'text', 'content': text}]


if __name__ == "__main__":
    # Smoke tests
    assert fix_umlaut_substitutions("oe und ae") == "ö und ä"
    assert normalize_unicode("Stra\u00dfe") == "Straße"
    result = correct_german_ocr("Fuer den Grenzwert gilt: oe kleiner ae")
    print(f"Corrected: {result}")
    print("german_postprocessing: all checks passed")
