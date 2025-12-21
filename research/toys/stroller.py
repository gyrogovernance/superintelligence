# scroller.py
import gradio as gr
import json
from pathlib import Path
from typing import List, Tuple
from baby.intelligence import AgentPool, orchestrate_turn

# Load preferences
PREFERENCES_PATH = Path("memories/memory_preferences.json")
with open(PREFERENCES_PATH) as f:
    PREFERENCES = json.load(f)

BASE_PATH = Path(PREFERENCES.get("base_path", PREFERENCES_PATH.parent)).resolve()

# Initialize shared AgentPool
agent_pool = AgentPool(
    ontology_path=PREFERENCES["ontology"]["ontology_map_path"],
    base_knowledge_path=PREFERENCES["public_knowledge"]["path"],
    preferences=PREFERENCES,
    allowed_ids={"user", "system", "assistant"},
    allow_auto_create=False,
    private_agents_base_path=str(BASE_PATH / PREFERENCES["private_knowledge"]["base_path"]),
    base_path=BASE_PATH,
)
agent_pool.ensure_triad()


# Process a chat turn
def chat_fn(message: str, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if not message:
        return history

    user_id = "user"
    assistant_id = "assistant"

    # Get system prompt if it exists
    system_prompt = ""
    try:
        system_prompt = system_box.value
    except AttributeError:
        pass

    # Handle system prompt on first turn
    if system_prompt and agent_pool.get(assistant_id).engine.cycle_count == 0:
        system_agent = agent_pool.get("system")
        from toys.communication import tokenizer as gyrotok

        stimulus = system_agent.respond(gyrotok.encode(system_prompt, name=PREFERENCES["tokenizer"]["name"]))
        agent_pool.get(assistant_id).ingest(stimulus)

    # Generate response
    try:
        reply = orchestrate_turn(
            agent_pool, user_id, assistant_id, message, tokenizer_name=PREFERENCES["tokenizer"]["name"]
        )
    except Exception as e:
        reply = f"Error: {str(e)}"

    # Update history
    history = history or []
    history.append((message, reply))
    return history


# Reset agents
def reset_agents() -> List[Tuple[str, str]]:
    agent_pool.close_all()
    agent_pool.ensure_triad()
    return []


# Get status
def get_status() -> str:
    try:
        assistant = agent_pool.get("assistant")
        cycles = assistant.engine.cycle_count
        return f"🍼 GyroSI Baby | Cycles: {cycles}"
    except (AttributeError, KeyError):
        return "🍼 GyroSI Baby | Initializing..."


# Create the interface with a light gray theme
with gr.Blocks(title="GyroSI Baby 🍼") as demo:

    gr.Markdown("# 🍼 Gyro Superintelligence Baby")
    status_bar = gr.Markdown(value=get_status())

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="Type your message here...", show_label=False)

            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Reset")

        with gr.Column(scale=1):
            system_box = gr.Textbox(placeholder="System prompt (optional)", label="System Prompt", lines=3)
            gr.Markdown("### Model Info")
            gr.Markdown(f"Tokenizer: {PREFERENCES['tokenizer']['name']}")

    # Event handlers
    submit_btn.click(chat_fn, [msg, chatbot], [chatbot], queue=False).then(lambda: "", None, msg).then(
        get_status, None, status_bar
    )

    msg.submit(chat_fn, [msg, chatbot], [chatbot], queue=False).then(lambda: "", None, msg).then(
        get_status, None, status_bar
    )

    clear_btn.click(reset_agents, None, chatbot).then(get_status, None, status_bar)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
