# GyroGem/chat.py
# Interactive chat with GyroGem Guard Instruct
# [Authority:Indirect] + [Agency:Indirect]

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Ensure package root is in path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from GyroGem.agent.context import GYROGEM_SYSTEM_PROMPT


def load_model():
    # Try local path first, then fall back to HF
    local_path = _root / "data" / "models" / "GyroGem-Guard-Instruct"
    model_id = str(local_path) if local_path.exists() else "gyrogovernance/gyrogem-guard-instruct"

    print(f"Loading from: {model_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        # Force float32
        for param in model.parameters():
            param.data = param.data.float()
        for buf in model.buffers():
            buf.data = buf.data.float()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return tokenizer, model, device

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def main():
    print("Initializing GyroGem Guard Instruct...")
    tokenizer, model, device = load_model()

    print("\n" + "="*60)
    print("GyroGem Guard Instruct - Interactive Mode")
    print(f"Device: {device.upper()}")
    print("Type 'quit', 'exit', or Ctrl+C to stop.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("Input > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                break

            # Construct input with system prompt (matching inference/training format)
            full_input = f"{GYROGEM_SYSTEM_PROMPT}\n\n{user_input}"

            inputs = tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)

            print("Thinking...", end="", flush=True)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,  # Large limit as requested
                    do_sample=False,      # Deterministic
                    num_beams=1,
                    use_cache=True
                )

            # Clear line
            print("\r" + " " * 20 + "\r", end="", flush=True)

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Output > {response}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
