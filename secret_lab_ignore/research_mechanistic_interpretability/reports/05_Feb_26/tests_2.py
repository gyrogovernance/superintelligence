import numpy as np
import torch
from transformers import AutoModelForCausalLM

delta_BU = 0.195342176580
Q_G = 4 * np.pi

def compute_coverage(wq, wk, head_dim):
    kernel = wq @ wk.T
    svs = torch.linalg.svdvals(kernel).numpy()
    svs_norm = svs / (svs.sum() + 1e-10)
    eff_dim = 1.0 / (np.sum(svs_norm**2) + 1e-10)
    return eff_dim / head_dim

def main():
    print("=== Verification: OLMo vs Random Baseline ===")
    print(f"CGM prediction: 32 layers × π/8 = 4π = {Q_G:.4f}")

    model = AutoModelForCausalLM.from_pretrained(
        "data/models/Olmo-3-7B-Instruct",
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True
    )

    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    hidden_dim = model.config.hidden_size

    # Collect OLMo coverages
    olmo_coverages = []
    with torch.no_grad():
        for layer in model.model.layers:
            wq = layer.self_attn.q_proj.weight.view(n_heads, head_dim, -1).float()
            wk = layer.self_attn.k_proj.weight.view(n_heads, head_dim, -1).float()
            for h in range(n_heads):
                olmo_coverages.append(compute_coverage(wq[h], wk[h], head_dim))

    olmo_coverages = np.array(olmo_coverages)
    olmo_sum = olmo_coverages.sum() / n_heads  # Sum over layers, mean over heads

    # Random baseline: Gaussian weights with same shape and scale
    print("\nGenerating random baselines (10 trials)...")
    random_sums = []

    for trial in range(10):
        torch.manual_seed(trial)
        trial_coverages = []

        for _ in range(32):  # 32 layers
            for _ in range(n_heads):
                # Match OLMo's weight scale approximately
                wq_rand = torch.randn(head_dim, hidden_dim) * 0.02
                wk_rand = torch.randn(head_dim, hidden_dim) * 0.02
                trial_coverages.append(compute_coverage(wq_rand, wk_rand, head_dim))

        trial_sum = np.sum(trial_coverages) / n_heads
        random_sums.append(trial_sum)

    random_sums = np.array(random_sums)

    print("\n=== Results ===")
    print(f"OLMo cumulative (per-head mean): {olmo_sum:.4f}")
    print(f"Random cumulative: {random_sums.mean():.4f} ± {random_sums.std():.4f}")
    print(f"Q_G = 4π: {Q_G:.4f}")

    # Statistical significance
    z_score = (olmo_sum - random_sums.mean()) / random_sums.std()
    print(f"\nZ-score (OLMo vs Random): {z_score:.2f}")

    # How close to 4π?
    olmo_error = abs(olmo_sum - Q_G) / Q_G
    random_error = abs(random_sums.mean() - Q_G) / Q_G
    print(f"OLMo error from 4π: {olmo_error:.4%}")
    print(f"Random error from 4π: {random_error:.4%}")

    # Per-layer analysis
    print("\n=== Per-Layer Means ===")
    layer_means = []
    for i in range(32):
        layer_cov = olmo_coverages[i*n_heads:(i+1)*n_heads]
        layer_means.append(layer_cov.mean())

    layer_means = np.array(layer_means)

    # Check if L0 is special
    print(f"L0:  {layer_means[0]:.4f} (δ_BU/2 = {delta_BU/2:.4f})")
    print(f"L1-31 mean: {layer_means[1:].mean():.4f} (π/8 = {np.pi/8:.4f})")

    # The formula: L0 + 31*(π/8) should ≈ 4π - π/8 + δ_BU/2
    predicted_sum = layer_means[0] + 31 * (np.pi/8)
    print(f"\nL0 + 31×(π/8) = {predicted_sum:.4f}")
    print(f"Actual sum: {layer_means.sum():.4f}")
    print(f"4π: {Q_G:.4f}")

    # What if L0 is the "aperture" correction?
    correction = layer_means[0] - np.pi/8
    print(f"\nL0 deviation from π/8: {correction:.4f}")
    print(f"This equals: {correction/delta_BU:.2f} × δ_BU")

if __name__ == "__main__":
    main()
