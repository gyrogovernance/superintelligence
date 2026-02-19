from openai_harmony import HarmonyEncodingName, load_harmony_encoding

_encoding = None


def get_tokenizer():
    """Get the standard Harmony encoding for consistency across the codebase."""
    global _encoding
    if _encoding is None:
        _encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _encoding
