# baby/constants/harmony_tokens.py

# Harmony control token constants - single source of truth
# Based on o200k_harmony encoding specification

# Core Harmony tokens
START = 200006  # <|start|>
CHANNEL = 200005  # <|channel|>
MESSAGE = 200008  # <|message|>
END = 200007  # <|end|>
RETURN = 200002  # <|return|>
CALL = 200012  # <|call|>

# Reserved tokens (not used in current implementation)
CONSTRAIN = 200003  # <|constrain|>
RESERVED_200000 = 200000  # <|reserved_200000|>
RESERVED_200001 = 200001  # <|reserved_200001|>

# Role constants (string values for harmony library compatibility)
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_DEVELOPER = "developer"
ROLE_TOOL = "tool"

# Channel constants
CHANNEL_FINAL = "final"
CHANNEL_ANALYSIS = "analysis"
CHANNEL_COMMENTARY = "commentary"


# Dynamic helpers for runtime token ID resolution
def token_id(encoding, text: str) -> int:
    """Get token ID for text from encoding, return first token or -1 if none."""
    ids = encoding.encode(text)
    return ids[0] if ids else -1


def assistant_role_id(encoding) -> int:
    """Get token ID for 'assistant' role."""
    return token_id(encoding, "assistant")


def final_channel_id(encoding) -> int:
    """Get token ID for 'final' channel."""
    return token_id(encoding, "final")


# Token sets for different use cases

# Control sets - single source of truth
GENERATION_EXCLUDED = {START, CHANNEL, MESSAGE, END, RETURN, CALL}  # content must never emit these

STATE_MACHINE_ONLY = {END, RETURN}  # only the backend framing should emit these

# All control tokens (for exclusion from orbit sweeps)
ALL_CONTROL_TOKENS = GENERATION_EXCLUDED  # single source of truth
