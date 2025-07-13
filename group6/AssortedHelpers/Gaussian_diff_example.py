import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

version_identifier = "_v4"
# Load your three samples
real = np.load(f'samples/g6_real_sample{version_identifier}.npy')      # shape (2500,)
noised = np.load(f'samples/g6_noised_sample{version_identifier}.npy')   # shape (2500,)
synthetic = np.load(f'samples/g6_synthetic_sample{version_identifier}.npy')  # shape (2500,)
synthetic *= 100
# Stack them into one array
X = np.vstack([real, noised, synthetic])

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
labels = ['Real', 'Noised', 'Synthetic']
colors = ['blue', 'orange', 'green']

# Plot each point
for i in range(3):
    plt.scatter(X_2d[i, 0], X_2d[i, 1], color=colors[i], label=labels[i], s=100)

# Draw arrows showing the progression
plt.arrow(X_2d[0,0], X_2d[0,1], X_2d[1,0]-X_2d[0,0], X_2d[1,1]-X_2d[0,1],
          length_includes_head=True, head_width=0.05, color='gray')
plt.arrow(X_2d[1,0], X_2d[1,1], X_2d[2,0]-X_2d[1,0], X_2d[2,1]-X_2d[1,1],
          length_includes_head=True, head_width=0.05, color='gray')

plt.legend()
plt.title('Diffusion Process: Real → Noised → Synthetic (PCA 2D)')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.grid(True)

# --- SAVE instead of SHOW ---
plt.savefig(f'images/diffusion/PCA_plot{version_identifier}.png', dpi=300)
print("Plot saved as PCA_plot{version_identifier}.png")
