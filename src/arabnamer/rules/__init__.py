"""Rule-based transliteration (verbatim from thesis validator.name_transliterator)."""
from .name_transliterator import transliterate_name, _translit_word

__all__ = ["transliterate_name", "_translit_word"]
