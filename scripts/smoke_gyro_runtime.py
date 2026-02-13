import os, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.getcwd())

from src.agent.runtime import GyroASI, GyroASIConfig


MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
ATLAS_DIR = "data/atlas"
ADAPTOR_PATH = "data/atlas/adaptor_olmo_k16_R16.npz"


def main():
    if not MODEL_DIR.exists():
        print(f"Error: {MODEL_DIR} not found.")
        return

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    embed = model.model.embed_tokens.weight.detach().float().cpu().numpy()  # [vocab,4096]
    lm_head = model.lm_head.weight.detach().float().cpu().numpy()  # [vocab,4096]
    norm_w = model.model.norm.weight.detach().float().cpu().numpy()  # [4096]
    d = embed.shape[1]
    assert d == 4096

    if not Path(ADAPTOR_PATH).exists():
        print(f"Error: {ADAPTOR_PATH} not found. Run scripts/build_adaptor_olmo_chart.py first.")
        return

    cfg = GyroASIConfig(
        atlas_dir=ATLAS_DIR,
        adaptor_path=ADAPTOR_PATH,
        K=16,
        eta=0.001,
        temperature=0.7,
        weight_penalty=0.02,
        op_blend=0.9,
        drive_blend=0.1,
        seed=0,
    )
    rt = GyroASI(cfg)

    prompt = "The purpose of good governance is"
    ids = tok.encode(prompt, add_special_tokens=False)

    microsteps = 4
    gen_tokens = 24

    # Warmup on prompt.
    for tid in ids:
        for _ in range(microsteps):
            rt.step_with_token_id(int(tid), embed)

    generated: list[int] = []
    gen_rng = np.random.default_rng(0)
    drive_tid = int(ids[-1]) if ids else 0
    for _ in range(gen_tokens):
        for ms in range(microsteps):
            if ms == 0:
                rt.step_with_token_id(drive_tid, embed)
            else:
                x_in = rt.adaptor.unchart_vec(rt.ctx.X).astype(np.float32)
                rt.step_with_semantic_vector(x_in)

        x_hat = rt.adaptor.unchart_vec(rt.ctx.X).astype(np.float32)  # [4096]
        # OLMo decode path uses final RMSNorm before lm_head.
        rms = float(np.sqrt(np.mean(x_hat * x_hat)) + 1e-6)
        x_norm = (x_hat / rms) * norm_w
        logits = lm_head @ x_norm  # exact measurement decoder

        if generated:
            for t in generated[-32:]:
                logits[int(t)] -= 1.5

        k = 40
        top_idx = np.argpartition(logits, -k)[-k:]
        top_logits = logits[top_idx]
        top_logits = top_logits - float(np.max(top_logits))
        p = np.exp(top_logits / 0.9).astype(np.float64)
        p /= float(np.sum(p) + 1e-18)
        drive_tid = int(top_idx[int(gen_rng.choice(len(top_idx), p=p))])
        generated.append(drive_tid)

    text = tok.decode(generated, clean_up_tokenization_spaces=False)

    print("Kernel signature:", rt.kernel.signature())
    print("Genealogy length:", len(rt.genealogy))
    bal = rt.roles.Bal(rt.sigma)
    print("Balanced projection max energy:", bal.max())
    print("Balanced projection sample (first 8):", bal[:8])
    print("Generated token ids:", generated)
    print("Generated text:", ascii(text))


if __name__ == "__main__":
    main()
