import numpy as np
import matplotlib.pyplot as plt

version_identifier = "_v2"
# Load your three samples
real = np.load(f'samples/g6_real_sample{version_identifier}.npy').squeeze()
noised = np.load(f'samples/g6_noised_sample{version_identifier}.npy').squeeze()
synthetic = np.load(f'samples/g6_synthetic_sample{version_identifier}.npy').squeeze()
#synthetic *= 25

# Feature index array for X-axis
features = np.arange(real.shape[0])

# Create the plot
plt.figure(figsize=(12, 6))

#plt.plot(features[::5], real[::5], label='Real', color='blue')
#plt.plot(features[::3], noised[::3], label='Noised', color='orange')
#plt.plot(features[::5], synthetic[::5], label='Synthetic', color='green')

#plots every 10 feature vector values
plt.plot(features[::10], real[::10], label='Real', color='blue')
#plt.plot(features[::10], noised[::10], label='Noised', color='orange')
plt.plot(features[::10], synthetic[::10], label='Synthetic', color='green')

#plt.title('Feature Value Comparison: Real vs Noised vs Synthetic', fontsize=14)
plt.title('Feature Value Comparison: Real vs Synthetic', fontsize=14)
plt.xlabel('Feature Index', fontsize=12)
plt.ylabel('Feature Value', fontsize=12)
plt.legend()
plt.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig(f'images/featureComp/feature_comparison_plot{version_identifier}-label.png', dpi=300)
print(f"Plot saved as feature_comparison_plot{version_identifier}.png")
