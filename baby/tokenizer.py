from openai_harmony import load_harmony_encoding, HarmonyEncodingName

_encoding = None


def get_tokenizer():
    """Get the standard Harmony encoding for consistency across the codebase."""
    global _encoding
    if _encoding is None:
        _encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _encoding
