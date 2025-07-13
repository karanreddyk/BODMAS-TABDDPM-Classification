import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

version_identifier = "_v7"

# Load and squeeze your three samples
#real = np.load(f'samples/g6_real_sample{version_identifier}.npy').squeeze()
#noised = np.load(f'samples/g6_noised_sample{version_identifier}.npy').squeeze()
#synthetic = np.load(f'samples/g6_synthetic_sample{version_identifier}.npy').squeeze()
#synthetic *= 100

real_npy = np.load(f'../BODMAS/code/multiple_data/g6data/un_X_train_real_seed0.npy')
real = real_npy[1]
synthetic_npy = np.load(f'exp/bodmas/ddpm_un_sf_tune_best/high_quality/X_num_train.npy')
synthetic = synthetic_npy[1]


# Normalize to probability distributions
def normalize_to_prob(x):
    x = x - np.min(x)  # Shift to make everything >= 0
    total = np.sum(x)
    if total == 0:
        return np.ones_like(x) / len(x)  # Avoid division by zero
    return x / total

# Normalize all samples
real_prob = normalize_to_prob(real)
#noised_prob = normalize_to_prob(noised)
synthetic_prob = normalize_to_prob(synthetic)

# Compute JS Divergence
#js_real_noised = jensenshannon(real_prob, noised_prob)**2
js_real_synthetic = jensenshannon(real_prob, synthetic_prob)**2

# Print the results
#print(f"JS Divergence (Real vs Noised): {js_real_noised:.6f}")
print(f"JS Divergence (Real vs Synthetic): {js_real_synthetic:.6f}")

# Plot the divergences
#divergences = [js_real_noised, js_real_synthetic]
divergences = [js_real_synthetic]
labels = ['Real vs Synthetic']

plt.figure(figsize=(8, 5))
plt.bar(labels, divergences, color=['orange', 'green'])
plt.title('Jensen-Shannon Divergence')
plt.ylabel('Divergence')
plt.ylim(0, 1)
plt.grid(axis='y')

# Save the plot
plt.tight_layout()
plt.savefig(f'images/jsd/js_divergence_plot{version_identifier}-norm.png', dpi=300)
print("Plot saved as js_divergence_plot.png")