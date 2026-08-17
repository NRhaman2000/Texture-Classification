"""
make_fourier_figure.py

Generates Figure 2 for the report: an illustration of how the Fourier
ring/wedge features work.

The report's Method section describes partitioning the 2D power spectrum into
radial rings (which capture SCALE) and angular wedges (which capture
ORIENTATION), but has no figure showing it. This produces that figure.

Run:  python make_fourier_figure.py   ->  fourier_partition.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#16283b"
INK2 = "#43566a"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

N_RINGS = 6
N_WEDGES = 6
SIZE = 128


def make_texture():
    """A synthetic oriented texture: a dominant diagonal grating plus a finer one."""
    x, y = np.meshgrid(np.linspace(0, 1, SIZE), np.linspace(0, 1, SIZE))
    theta = np.pi / 5
    t = np.sin(2 * np.pi * 9 * (x * np.cos(theta) + y * np.sin(theta)))
    t += 0.45 * np.sin(2 * np.pi * 22 * x)
    return np.clip((t - t.min()) / (np.ptp(t) + 1e-9), 0, 1)


def polar_grids(shape):
    h, w = shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r = r / (r.max() + 1e-9)
    th = np.arctan2(Y - cy, X - cx) % np.pi
    return r, th


def main():
    img = make_texture()
    P = np.fft.fftshift(np.abs(np.fft.fft2(img)) ** 2)
    logP = np.log1p(P)
    r, th = polar_grids(P.shape)

    ring_id = np.clip((r * N_RINGS).astype(int), 0, N_RINGS - 1)
    wedge_id = np.clip((th / np.pi * N_WEDGES).astype(int), 0, N_WEDGES - 1)

    # the actual feature values
    ring_energy = np.array([P[ring_id == i].sum() for i in range(N_RINGS)])
    wedge_energy = np.array([P[wedge_id == j].sum() for j in range(N_WEDGES)])
    ring_energy /= ring_energy.sum()
    wedge_energy /= wedge_energy.sum()

    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.3))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("(a) input texture", fontsize=10.5)

    axes[1].imshow(logP, cmap="magma")
    axes[1].set_title("(b) log power spectrum", fontsize=10.5)

    axes[2].imshow(ring_id, cmap="viridis")
    axes[2].set_title(f"(c) {N_RINGS} radial rings = scale", fontsize=10.5)

    axes[3].imshow(wedge_id, cmap="twilight")
    axes[3].set_title(f"(d) {N_WEDGES} angular wedges = orientation", fontsize=10.5)

    for a in axes[:4]:
        a.set_xticks([]); a.set_yticks([])

    # the resulting 12-dimensional feature vector
    ax = axes[4]
    idx = np.arange(N_RINGS + N_WEDGES)
    vals = np.concatenate([ring_energy, wedge_energy])
    colors = ["#0f766e"] * N_RINGS + ["#b45309"] * N_WEDGES
    ax.bar(idx, vals, color=colors, width=0.72)
    ax.axvline(N_RINGS - 0.5, color=INK2, ls="--", lw=1.1)
    ax.text(N_RINGS / 2 - 0.5, max(vals) * 1.04, "rings", ha="center",
            fontsize=9, color="#0f766e", weight="bold")
    ax.text(N_RINGS + N_WEDGES / 2 - 0.5, max(vals) * 1.04, "wedges", ha="center",
            fontsize=9, color="#b45309", weight="bold")
    ax.set_title("(e) resulting 12-D feature vector", fontsize=10.5)
    ax.set_xticks([]); ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("normalized energy", fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    fig.suptitle("Fourier filter-bank features: partitioning the power spectrum",
                 fontsize=12.5, weight="bold", y=1.04)
    fig.text(0.5, -0.06,
             "Distance from the spectral centre corresponds to spatial frequency, so ring energies describe texture scale. "
             "Angle corresponds to\npattern direction, so wedge energies describe orientation. "
             "The dominant diagonal grating in (a) appears as the bright off-centre lobes in (b).",
             ha="center", fontsize=9, style="italic", color=INK2)

    plt.tight_layout()
    plt.savefig("fourier_partition.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fourier_partition.png")
    print(f"ring energies : {np.round(ring_energy, 4)}")
    print(f"wedge energies: {np.round(wedge_energy, 4)}")


if __name__ == "__main__":
    main()
