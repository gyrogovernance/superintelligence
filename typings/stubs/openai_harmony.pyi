# Type stubs for openai_harmony module
from typing import Any, List, Optional, Union, Literal
from dataclasses import dataclass

# String literal types for better compatibility
HarmonyEncodingNameType = Literal["harmony-gpt-oss"]
ReasoningEffortType = Literal["low", "medium", "high"]
RoleType = Literal["system", "user", "assistant", "developer"]

# Enum-like class for HarmonyEncodingName
class HarmonyEncodingName:
    HARMONY_GPT_OSS: str = "harmony-gpt-oss"

# Enum-like classes for attribute access
class Role:
    SYSTEM: "Role"
    USER: "Role"
    ASSISTANT: "Role"
    DEVELOPER: "Role"
    TOOL: "Role"

    def __init__(self, value: str) -> None: ...
    @property
    def value(self) -> str: ...

class Author:
    SYSTEM: str = "system"
    USER: str = "user"
    ASSISTANT: str = "assistant"
    DEVELOPER: str = "developer"
    TOOL: str = "tool"

    @classmethod
    def new(cls, role: Union[Role, RoleType, str], recipient: Optional[str] = None) -> "Author": ...

class ReasoningEffort:
    HIGH: str = "high"
    MEDIUM: str = "medium"
    LOW: str = "low"

@dataclass
class RenderConversationConfig:
    """Configuration for rendering conversations in harmony encoding."""

    include_reasoning: bool = True
    include_channels: bool = True

class HarmonyEncoding:
    def __init__(self) -> None: ...
    def render_conversation_for_completion(
        self, conversation: Conversation, next_turn_role: Role, config: RenderConversationConfig | None = None
    ) -> list[int]: ...
    def render_conversation_for_training(
        self, conversation: Conversation, config: RenderConversationConfig | None = None
    ) -> list[int]: ...
    def render_conversation(
        self, conversation: Conversation, config: RenderConversationConfig | None = None
    ) -> list[int]: ...
    def render(self, message: Message) -> list[int]: ...
    def parse_messages_from_completion_tokens(self, tokens: list[int], role: Role | None = None) -> list[Message]: ...
    def decode(self, tokens: list[int]) -> str: ...
    def encode(self, text: str, allowed_special: str = "none") -> list[int]: ...
    def stop_tokens(self) -> list[int]: ...
    def stop_tokens_for_assistant_actions(self) -> list[int]: ...

class StreamState:
    STREAMING: str = "streaming"
    COMPLETE: str = "complete"
    ERROR: str = "error"
    EXPECT_START: str = "expect_start"

class SystemContent:
    @classmethod
    def new(cls) -> "SystemContent": ...
    def with_model_identity(self, identity: str) -> "SystemContent": ...
    def with_reasoning_effort(self, effort: Union[ReasoningEffort, str]) -> "SystemContent": ...
    def with_conversation_start_date(self, date: str) -> "SystemContent": ...
    def with_knowledge_cutoff(self, cutoff: str) -> "SystemContent": ...
    def with_required_channels(self, channels: List[str]) -> "SystemContent": ...
    def with_tools(self, tools: List[Any]) -> "SystemContent": ...
    def __getitem__(self, key: Any) -> Any: ...
    @property
    def model_identity(self) -> str: ...
    @property
    def reasoning_effort(self) -> str: ...
    @property
    def conversation_start_date(self) -> str: ...
    @property
    def knowledge_cutoff(self) -> str: ...

class DeveloperContent:
    text: str
    instructions: str
    def __init__(self, text: str) -> None: ...
    @classmethod
    def new(cls, content: str) -> "DeveloperContent": ...
    def __getattr__(self, name: str) -> Any: ...

class ToolDescription:
    name: str
    description: str
    def __init__(self, name: str, description: str) -> None: ...
    @classmethod
    def new(cls, name: str, description: str) -> "ToolDescription": ...
    def __getattr__(self, name: str) -> Any: ...

class TextContent:
    text: str
    def __init__(self, text: str) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class Message:
    role: Union[str, RoleType]
    content: Union[str, List[TextContent], List[Any], SystemContent, Any]
    channel: Optional[str]
    author: Optional[Author]
    recipient: Optional[str]

    def __init__(
        self, author: Optional[Author] = None, content: Union[str, List[TextContent], List[Any]] = None, **kwargs: Any
    ) -> None: ...
    @classmethod
    def from_role_and_content(
        cls, role: Union[Role, RoleType, str], content: Union[str, SystemContent, DeveloperContent]
    ) -> "Message": ...
    @classmethod
    def from_author_and_content(
        cls, author: Author, content: Union[str, SystemContent, DeveloperContent]
    ) -> "Message": ...
    def with_recipient(self, recipient: str) -> "Message": ...
    def with_channel(self, channel: str) -> "Message": ...
    def __getattr__(self, name: str) -> Any: ...

class Conversation:
    messages: List[Message]
    def __init__(self, messages: List[Message]) -> None: ...
    @classmethod
    def from_messages(cls, messages: List[Message]) -> "Conversation": ...
    def to_tokens(self, encoding: Any) -> List[int]: ...

class StreamableParser:
    current_role: Optional[RoleType]
    current_channel: Optional[str]
    last_content_delta: str
    current_content_type: str
    current_recipient: Optional[str]
    current_content: str
    state: str
    messages: List[Message]

    def __init__(self, encoding: Any, role: Union[Role, RoleType, str]) -> None: ...
    def process(self, token: int) -> None: ...
    def parse_tokens(self, tokens: List[int]) -> List[Message]: ...

def load_harmony_encoding(name: Union[HarmonyEncodingNameType, str]) -> Any: ...
