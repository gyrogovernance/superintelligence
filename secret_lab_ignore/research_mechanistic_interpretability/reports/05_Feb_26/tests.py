import numpy as np
import torch
from transformers import AutoModelForCausalLM

delta_BU = 0.195342176580
m_a = 0.199471140201
Q_G = 4 * np.pi

def main():
    print("=== Head-Level Coverage Distribution ===")
    print(f"CGM targets: δ_BU/2={delta_BU/2:.4f}, π/8={np.pi/8:.4f}, 2δ_BU={2*delta_BU:.4f}")

    model = AutoModelForCausalLM.from_pretrained(
        "data/models/Olmo-3-7B-Instruct",
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True
    )

    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    all_coverages = []
    l0_coverages = []

    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            wq = layer.self_attn.q_proj.weight.view(n_heads, head_dim, -1).float()
            wk = layer.self_attn.k_proj.weight.view(n_heads, head_dim, -1).float()

            for h in range(n_heads):
                kernel = wq[h] @ wk[h].T
                svs = torch.linalg.svdvals(kernel).numpy()
                svs_norm = svs / (svs.sum() + 1e-10)
                eff_dim = 1.0 / (np.sum(svs_norm**2) + 1e-10)
                coverage = eff_dim / head_dim

                all_coverages.append((i, h, coverage))
                if i == 0:
                    l0_coverages.append(coverage)

    all_cov = np.array([c[2] for c in all_coverages])
    l0_cov = np.array(l0_coverages)
    other_cov = np.array([c[2] for c in all_coverages if c[0] > 0])

    print(f"\n=== Layer 0 Analysis (n={len(l0_cov)}) ===")
    print(f"L0 mean: {l0_cov.mean():.4f} ± {l0_cov.std():.4f}")
    print(f"L0 min:  {l0_cov.min():.4f}, max: {l0_cov.max():.4f}")
    print(f"Target δ_BU/2: {delta_BU/2:.4f}")
    print(f"Heads within 10% of δ_BU/2: {np.sum(np.abs(l0_cov - delta_BU/2) < 0.01)}/{len(l0_cov)}")

    # Statistical test: is L0 different from others?
    t_stat = (l0_cov.mean() - other_cov.mean()) / np.sqrt(l0_cov.var()/len(l0_cov) + other_cov.var()/len(other_cov))
    print(f"L0 vs Others t-stat: {t_stat:.2f}")

    print(f"\n=== All Layers (n={len(all_cov)}) ===")
    print(f"Global mean: {all_cov.mean():.4f}")
    print(f"π/8 = {np.pi/8:.4f}, 2δ_BU = {2*delta_BU:.4f}")

    # Histogram around CGM values
    bins_delta = [0, delta_BU/2*0.8, delta_BU/2*1.2, np.pi/8*0.9, np.pi/8*1.1, 0.6]
    hist, _ = np.histogram(all_cov, bins=bins_delta)
    print("\nDistribution:")
    print(f"  <0.8×(δ_BU/2): {hist[0]}")
    print(f"  ≈δ_BU/2 (±20%): {hist[1]}")
    print(f"  middle: {hist[2]}")
    print(f"  ≈π/8 (±10%): {hist[3]}")
    print(f"  >1.1×(π/8): {hist[4]}")

    # Per-layer means with CGM comparison
    print("\n=== Layer Progression ===")
    layer_means = []
    for i in range(32):
        layer_cov = [c[2] for c in all_coverages if c[0] == i]
        layer_means.append(np.mean(layer_cov))

    # Check depth-4 structure: do layers 0,4,8,... or 0,8,16,24 show patterns?
    print("Every 8th layer (potential depth-4 cycle):")
    for i in [0, 8, 16, 24]:
        print(f"  L{i:02d}: {layer_means[i]:.4f}")

    # Cumulative effect
    cumsum = np.cumsum(layer_means)
    print("\nCumulative coverage at key layers:")
    print(f"  After L7:  {cumsum[7]:.4f} (×δ_BU: {cumsum[7]/delta_BU:.2f})")
    print(f"  After L15: {cumsum[15]:.4f} (×δ_BU: {cumsum[15]/delta_BU:.2f})")
    print(f"  After L23: {cumsum[23]:.4f} (×δ_BU: {cumsum[23]/delta_BU:.2f})")
    print(f"  After L31: {cumsum[31]:.4f} (×δ_BU: {cumsum[31]/delta_BU:.2f})")
    print(f"  Total/32 = mean: {cumsum[31]/32:.4f}")

if __name__ == "__main__":
    main()
