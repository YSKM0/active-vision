import numpy as np
import matplotlib.pyplot as plt

reinforce = False

# Load multiple eigenvalue sets
eigvals_rs = np.load("/local/home/hanwliu/activevision/eigenvalues_rs_gcd_[5, 30].npy")
eigvals_fvs = np.load(
    "/local/home/hanwliu/activevision/eigenvalues_fvs_gcd_[5, 30].npy"
)
eigvals_vlm = np.load(
    "/local/home/hanwliu/activevision/eigenvalues_vlm_gcd_[5, 30].npy"
)
eigvals_fvsvlm = np.load(
    "/local/home/hanwliu/activevision/eigenvalues_fvs_vlm_gcd_[5, 30].npy"
)

if reinforce:
    eigvals_rs = eigvals_rs**2
    eigvals_fvs = eigvals_fvs**2
    eigvals_vlm = eigvals_vlm**2
    eigvals_fvsvlm = eigvals_fvsvlm**2
    plot_label = r"$\lambda_i^2$"
    plt_title = "Laplacian Spectrum² Comparison"
else:
    plot_label = r"$\lambda_i$"
    plt_title = "Laplacian Spectrum Comparison"

# Plot all on the same figure
plt.figure()
plt.plot(eigvals_rs, label="rs", color="blue", linewidth=2)
plt.plot(eigvals_fvs, label="fvs", color="orange", linewidth=2)
plt.plot(eigvals_vlm, label="vlm", color="green", linewidth=2)
plt.plot(eigvals_fvsvlm, label="fvs_vlm", color="red", linewidth=2)

plt.yscale("linear")
plt.title(plt_title)
plt.xlabel("Eigenvalue Index")
plt.ylabel(plot_label)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
