from typing import Any, Literal

# ReasoningEffort import removed as it was unused
from pydantic import BaseModel

MODEL_IDENTIFIER = "gyrosi-babylm"
DEFAULT_TEMPERATURE = 0.0
REASONING_EFFORT = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 10_000


class UrlCitation(BaseModel):
    type: Literal["url_citation"]
    end_index: int
    start_index: int
    url: str
    title: str


class TextContentItem(BaseModel):
    type: Literal["text"] | Literal["input_text"] | Literal["output_text"]
    text: str
    status: str | None = "completed"
    annotations: list[UrlCitation] | None = None


class SummaryTextContentItem(BaseModel):
    # using summary for compatibility with the existing API
    type: Literal["summary_text"]
    text: str


class ReasoningTextContentItem(BaseModel):
    type: Literal["reasoning_text"]
    text: str


class ReasoningItem(BaseModel):
    id: str = "rs_1234"
    type: Literal["reasoning"]
    summary: list[SummaryTextContentItem]
    content: list[ReasoningTextContentItem] | None = []


class Item(BaseModel):
    type: Literal["message"] | None = "message"
    role: Literal["user", "assistant", "system"]
    content: list[TextContentItem] | str
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class FunctionCallItem(BaseModel):
    type: Literal["function_call"]
    name: str
    arguments: str
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    id: str = "fc_1234"
    call_id: str = "call_1234"


class FunctionCallOutputItem(BaseModel):
    type: Literal["function_call_output"]
    call_id: str = "call_1234"
    output: str


class WebSearchActionSearch(BaseModel):
    type: Literal["search"]
    query: str | None = None


class WebSearchActionOpenPage(BaseModel):
    type: Literal["open_page"]
    url: str | None = None


class WebSearchActionFind(BaseModel):
    type: Literal["find"]
    pattern: str | None = None
    url: str | None = None


class WebSearchCallItem(BaseModel):
    type: Literal["web_search_call"]
    id: str = "ws_1234"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    action: WebSearchActionSearch | WebSearchActionOpenPage | WebSearchActionFind


class Error(BaseModel):
    code: str
    message: str


class IncompleteDetails(BaseModel):
    reason: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class FunctionToolDefinition(BaseModel):
    type: Literal["function"]
    name: str
    parameters: dict[str, Any]  # this should be typed stricter if you add strict mode
    strict: bool = False  # change this if you support strict mode
    description: str | None = ""


class BrowserToolConfig(BaseModel):
    type: Literal["browser_search"]


class ReasoningConfig(BaseModel):
    effort: Literal["low", "medium", "high"] = REASONING_EFFORT


class ResponsesRequest(BaseModel):
    instructions: str | None = None
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS
    input: str | list[Item | ReasoningItem | FunctionCallItem | FunctionCallOutputItem | WebSearchCallItem]
    model: str | None = MODEL_IDENTIFIER
    stream: bool | None = False
    tools: list[FunctionToolDefinition | BrowserToolConfig] | None = []
    reasoning: ReasoningConfig | None = ReasoningConfig()
    metadata: dict[str, Any] | None = {}
    tool_choice: Literal["auto", "none"] | None = "auto"
    parallel_tool_calls: bool | None = False
    store: bool | None = False
    previous_response_id: str | None = None
    temperature: float | None = DEFAULT_TEMPERATURE
    include: list[str] | None = None


class ResponseObject(BaseModel):
    output: list[Item | ReasoningItem | FunctionCallItem | FunctionCallOutputItem | WebSearchCallItem]
    created_at: int
    usage: Usage | None = None
    status: Literal["completed", "failed", "incomplete", "in_progress"] = "in_progress"
    background: None = None
    error: Error | None = None
    incomplete_details: IncompleteDetails | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: dict[str, Any] | None = {}
    model: str | None = MODEL_IDENTIFIER
    parallel_tool_calls: bool | None = False
    previous_response_id: str | None = None
    id: str | None = "resp_1234"
    object: str | None = "response"
    text: dict[str, Any] | None = None
    tool_choice: str | None = "auto"
    top_p: int | None = 1
