#!/usr/bin/env python3
"""Interactive, simplified BrainWeb spin-echo contrast model with gadolinium.

Install: pip install numpy matplotlib nibabel
Run:     python mri_brainweb_contrast.py

Download the BrainWeb *discrete* MINC phantom and set BRAINWEB_MINC below.
Each tissue uses its assigned baseline proton-density, T1, and T2 values;
gadolinium is then applied uniformly to all non-background tissue labels. The displayed
signal uses the common short-TE ideal spin-echo approximation, not a complete
MRI acquisition or reconstruction model.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import nibabel as nib
import numpy as np

# Illustrative relaxivities in s^-1 mM^-1. Relaxivity depends on the specific
# Gd agent, field strength, temperature, and local microenvironment.
R1_RELAXIVITY = 4.5
R2_RELAXIVITY = 5.0

DEFAULT_TR_MS = 600.0
DEFAULT_TE_MS = 15.0
MIN_TE_MS = 5.0
SEQUENCE_TIMING_MARGIN_MS = 5.0
MAX_TE_MS = 300.0
N = 128
BRAINWEB_MINC = Path("phantom_1.0mm_normal_crisp.mnc")
SLICE_INDEX = 90

# BrainWeb discrete-phantom labels with illustrative nominal tissue properties:
# (display name, relative proton density, T1 seconds, T2 seconds). These are
# not quantitative parameter maps supplied by BrainWeb and should not be
# treated as scanner- or subject-specific values.
BRAINWEB_TISSUES = {
    0: ("Background",        0.00, 1.00, 0.001),
    1: ("CSF",               1.00, 4.00, 0.500),
    2: ("Gray matter",       0.86, 1.30, 0.050),
    3: ("White matter",      0.77, 0.83, 0.040),
    4: ("Fat",               0.90, 0.35, 0.060),
    5: ("Muscle",            0.80, 0.90, 0.035),
    6: ("Skin",              0.70, 0.60, 0.030),
    7: ("Skull",             0.05, 0.30, 0.010),
    8: ("Glial matter",      0.85, 1.10, 0.045),
    9: ("Connective tissue", 0.65, 0.90, 0.030),
}


def spin_echo_signal(t1_s, t2_s, tr_ms, te_ms):
    """Return the short-TE ideal spin-echo approximation, per unit proton density.

    It evaluates (1 - exp(-TR/T1)) exp(-TE/T2), assuming ideal 90/180-degree
    pulses, full refocusing of static dephasing, and TE small compared with TR.
    """
    tr_s = tr_ms / 1000
    te_s = te_ms / 1000
    return (1 - np.exp(-tr_s / t1_s)) * np.exp(-te_s / t2_s)


def load_brainweb_slice(filename, output_size, slice_index):
    """Load a BrainWeb axial label slice and resample labels without mixing them."""
    if not filename.is_file():
        raise FileNotFoundError(
            f"Cannot find {filename}. Set BRAINWEB_MINC to your downloaded .mnc file."
        )
    volume = np.asanyarray(nib.load(str(filename)).dataobj)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D MINC volume, got shape {volume.shape}")
    if not 0 <= slice_index < volume.shape[0]:
        raise ValueError(
            f"SLICE_INDEX must be between 0 and {volume.shape[0] - 1}."
        )

    # This phantom is expected to store its spatial axes as zspace, yspace,
    # xspace; confirm the orientation with the source metadata if another MINC
    # file is substituted.
    labels_yx = volume[slice_index].astype(np.uint8)

    # For the expected 1-mm phantom (217 x 181 in plane), center-pad the short
    # dimension with background before nearest-neighbor resampling. This
    # preserves labels, but it changes the displayed sampling grid.
    side = max(labels_yx.shape)
    padded = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - labels_yx.shape[0]) // 2
    x0 = (side - labels_yx.shape[1]) // 2
    padded[y0:y0 + labels_yx.shape[0], x0:x0 + labels_yx.shape[1]] = labels_yx
    sample = np.rint(np.linspace(0, side - 1, output_size)).astype(int)
    return padded[np.ix_(sample, sample)]


def brainweb_parameter_maps(labels):
    """Create proton-density, T1 and T2 maps from BrainWeb tissue labels."""
    unknown = set(np.unique(labels)) - set(BRAINWEB_TISSUES)
    if unknown:
        raise ValueError(f"Unknown BrainWeb labels: {sorted(unknown)}")

    rho = np.zeros(labels.shape, dtype=float)
    t1 = np.ones(labels.shape, dtype=float)
    t2 = np.full(labels.shape, 0.001, dtype=float)
    for label, (_, density, tissue_t1, tissue_t2) in BRAINWEB_TISSUES.items():
        mask = labels == label
        rho[mask] = density
        t1[mask] = tissue_t1
        t2[mask] = tissue_t2
    return rho, t1, t2


brainweb_labels = load_brainweb_slice(BRAINWEB_MINC, N, SLICE_INDEX)
rho_map, native_t1_map, native_t2_map = brainweb_parameter_maps(brainweb_labels)

fig, axes = plt.subplots(1, 4, figsize=(19, 6))
plt.subplots_adjust(left=0.055, right=0.985, bottom=0.47, wspace=0.34)
ax_image, ax_t1, ax_t2, ax_signal = axes

recovery_ms = np.linspace(0, 3000, 500)
echo_ms = np.linspace(0, 300, 500)
dose_points = np.linspace(0, 10, 300)

# The signal is normalized: relative PD and both relaxation factors are at most
# one. Keep this 0--1 display range fixed so image contrast changes without
# moving the grayscale colorbar's numeric scale.
signal_image = ax_image.imshow(
    np.zeros_like(rho_map), cmap="gray", origin="upper", vmin=0, vmax=1
)
ax_image.set(
    title=f"BrainWeb spin-echo image (z slice {SLICE_INDEX})",
    xlabel="Horizontal position (pixel)",
    ylabel="Vertical position (pixel)",
)
fig.colorbar(signal_image, ax=ax_image, fraction=.046, pad=.04,
             label="Relative signal")

t1_before_line, = ax_t1.plot([], [], "--", color="gray", label="Before Gd")
t1_after_line, = ax_t1.plot([], [], color="firebrick", lw=2.5, label="With Gd")
t1_tr_marker = ax_t1.axvline(DEFAULT_TR_MS, color="black", lw=1, ls=":", label="TR")
ax_t1.set(title="T1 longitudinal recovery", xlabel="Time after excitation (ms)",
          ylabel="Longitudinal magnetization (relative)", xlim=(0, 3000), ylim=(0, 1.04))
ax_t1.legend(loc="lower right")

t2_before_line, = ax_t2.plot([], [], "--", color="gray", label="Before Gd")
t2_after_line, = ax_t2.plot([], [], color="steelblue", lw=2.5, label="With Gd")
t2_te_marker = ax_t2.axvline(DEFAULT_TE_MS, color="black", lw=1, ls=":", label="TE")
ax_t2.set(title="T2 transverse decay", xlabel="Echo time (ms)",
          ylabel="Transverse magnetization (relative)", xlim=(0, 300), ylim=(0, 1.04))
ax_t2.legend(loc="upper right")

signal_line, = ax_signal.plot([], [], color="forestgreen", lw=2.5)
marker, = ax_signal.plot([], [], "o", color="darkorange", ms=9)
ax_signal.set(title="Spin-echo signal", xlabel="Local gadolinium concentration (mM)",
              ylabel="Relative signal", xlim=(0, 10), ylim=(0, 1.05))
info = ax_signal.text(0.03, 0.97, "", transform=ax_signal.transAxes, va="top",
                      bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})

# Each material retains its assigned baseline map. The dose control applies one
# Gd concentration to every non-background label and does not override baseline
# material properties. TR and TE are global acquisition settings.
dose_slider = Slider(fig.add_axes([0.19, 0.38, 0.62, 0.022]),
                     "Local Gd concentration (mM)", 0, 10, valinit=1.0, valstep=0.05)
tr_slider = Slider(fig.add_axes([0.19, 0.335, 0.62, 0.022]),
                   "TR (ms)", 50, 3000, valinit=DEFAULT_TR_MS, valstep=10)
te_slider = Slider(fig.add_axes([0.19, 0.29, 0.62, 0.022]),
                   "TE (ms)", MIN_TE_MS, MAX_TE_MS,
                   valinit=DEFAULT_TE_MS, valstep=1)
reset_button = Button(fig.add_axes([0.43, 0.20, 0.14, 0.045]), "Reset")


def update(_=None):
    concentration = dose_slider.val
    # Use gray matter only as a fixed reference for the relaxation and signal
    # plots; the image calculation below applies Gd to every tissue label.
    _, _, baseline_t1_s, baseline_t2_s = BRAINWEB_TISSUES[2]
    tr_ms = tr_slider.val
    # A single echo and its readout must finish before the next excitation.
    # Constrain TE in the widget itself, reserving a small generic timing
    # margin. Exact hardware limits depend on the scanner and pulse sequence.
    max_te_ms = min(MAX_TE_MS, tr_ms - SEQUENCE_TIMING_MARGIN_MS)
    te_slider.valmax = max_te_ms
    te_slider.ax.set_xlim(MIN_TE_MS, max_te_ms)
    if te_slider.val > max_te_ms:
        te_slider.set_val(max_te_ms)
        return
    te_ms = te_slider.val

    baseline_r1 = 1 / baseline_t1_s
    baseline_r2 = 1 / baseline_t2_s
    t1_after_s = 1 / (baseline_r1 + R1_RELAXIVITY * concentration)
    t2_after_s = 1 / (baseline_r2 + R2_RELAXIVITY * concentration)

    # The curves show normalized Bloch relaxation for the selected tissue.
    t1_before_line.set_data(recovery_ms, 1 - np.exp(-(recovery_ms / 1000) / baseline_t1_s))
    t1_after_line.set_data(recovery_ms, 1 - np.exp(-(recovery_ms / 1000) / t1_after_s))
    t1_tr_marker.set_xdata([tr_ms, tr_ms])
    t2_before_line.set_data(echo_ms, np.exp(-(echo_ms / 1000) / baseline_t2_s))
    t2_after_line.set_data(echo_ms, np.exp(-(echo_ms / 1000) / t2_after_s))
    t2_te_marker.set_xdata([te_ms, te_ms])

    t1_curve_s = 1 / (baseline_r1 + R1_RELAXIVITY * dose_points)
    t2_curve_s = 1 / (baseline_r2 + R2_RELAXIVITY * dose_points)
    signal_curve = spin_echo_signal(t1_curve_s, t2_curve_s, tr_ms, te_ms)
    current_signal = spin_echo_signal(t1_after_s, t2_after_s, tr_ms, te_ms)
    signal_line.set_data(dose_points, signal_curve)
    marker.set_data([concentration], [current_signal])
    ax_signal.set_title(
        f"Spin-echo signal: gray-matter reference (TR={tr_ms:.0f} ms, TE={te_ms:.0f} ms)"
    )

    # Apply one uniform concentration to every non-background label. This is
    # not a pharmacokinetic concentration model; each tissue begins from its
    # own baseline map and therefore retains tissue-dependent relaxation.
    tissue_mask = brainweb_labels != 0
    t1_map = native_t1_map.copy()
    t2_map = native_t2_map.copy()
    t1_map[tissue_mask] = 1 / (1 / native_t1_map[tissue_mask] + R1_RELAXIVITY * concentration)
    t2_map[tissue_mask] = 1 / (1 / native_t2_map[tissue_mask] + R2_RELAXIVITY * concentration)
    image = rho_map * spin_echo_signal(t1_map, t2_map, tr_ms, te_ms)
    signal_image.set_data(image)
    ax_image.set_title(f"BrainWeb spin-echo image: uniform Gd\n(z slice {SLICE_INDEX})")

    info.set_text(f"Uniform Gd: {concentration:.2f} mM\n"
                  "Plot reference: gray matter\n"
                  f"Effective T1: {t1_after_s * 1000:.0f} ms\n"
                  f"Effective T2: {t2_after_s * 1000:.1f} ms\n"
                  f"Reference signal: {current_signal:.3f}")
    fig.canvas.draw_idle()


def reset(_):
    for slider in (dose_slider, tr_slider, te_slider):
        slider.reset()


for slider in (dose_slider, tr_slider, te_slider):
    slider.on_changed(update)
reset_button.on_clicked(reset)
update()
plt.show()
