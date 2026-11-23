#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 00:11:44 2026

@author: ashlyn.nguyen
"""

"""
Interactive, ideal spin-echo MRI model.

This program makes a simple 64 × 64 pixel phantom containing four tissue
regions: air, CSF, fat, and muscle. Sliders change TR, TE, and each tissue's
T1/T2 values.

The model uses the spin-echo signal equation:

    S = rho × (1 - exp(-TR / T1)) × exp(-TE / T2)

where:
    S   = MRI signal magnitude
    rho = relative proton density
    TR  = repetition time
    TE  = echo time
    T1  = longitudinal relaxation time
    T2  = transverse relaxation time

This is an ideal model: it has no receiver noise, motion, or artifacts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# Number of pixels in both the horizontal and vertical image directions.
N = 64

# Each tissue is stored as:
#     (relative proton density rho, T1 in seconds, T2 in seconds, T2* in seconds)
#
# T2* is included for reference but is not used here because this is a
# spin-echo model. The 180-degree refocusing pulse in spin echo largely
# corrects for the extra magnetic-field dephasing represented by T2*.
tissues = {
    "Air": (0.0, 1.0, 0.001, 0.001),
    "CSF": (1.0, 4.0, 1.50, 0.50),
    "Fat": (0.9, 0.35, 0.080, 0.060),
    "Muscle": (0.8, 0.90, 0.050, 0.035),
}

# Create one N × N array for proton density and one each for T1 and T2.
rho = np.zeros((N, N))
t1 = np.empty((N, N))
t2 = np.empty((N, N))

# Divide the image into four equal quadrants:
#
#       Air  |  CSF
#      ------|------
#       Fat  | Muscle
#
half = N // 2

quadrants = {
    "Air": (slice(None, half), slice(None, half)),
    "CSF": (slice(None, half), slice(half, None)),
    "Fat": (slice(half, None), slice(None, half)),
    "Muscle": (slice(half, None), slice(half, None)),
}

# Fill every pixel in each quadrant with that tissue's rho, T1, and T2 values.
for name, region in quadrants.items():
    density, tissue_t1, tissue_t2, _ = tissues[name]

    rho[region] = density
    t1[region] = tissue_t1
    t2[region] = tissue_t2

# Create two plots:
#
# Left:  reconstructed MRI image
# Right: magnitude across the center row of the image
#
fig, ax = plt.subplots(1, 2, figsize=(13, 9))

# Leave space below the plots for sliders and the Reset button.
plt.subplots_adjust(
    left=0.08,
    right=0.96,
    top=0.93,
    bottom=0.46,
    wspace=0.32,
)

# Left plot: image display.
# vmin and vmax fix the color scale between 0 and 1.
clean_image = ax[0].imshow(
    np.zeros_like(rho),
    cmap="gray",
    vmin=0,
    vmax=1,
)

ax[0].set_title("Spin-echo reconstruction")
ax[0].set_xlabel("Horizontal pixel")
ax[0].set_ylabel("Vertical pixel")


# Write tissue names on the displayed image.
for x_label, y_label, label in [
    (N * 0.25, N * 0.25, "Air"),
    (N * 0.75, N * 0.25, "CSF"),
    (N * 0.25, N * 0.75, "Fat"),
    (N * 0.75, N * 0.75, "Muscle"),
]:
    ax[0].text(
        x_label,
        y_label,
        label,
        color="cyan",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

# Right plot: graph of the reconstructed magnitude along the center row.
ax[1].set_title("Centre-row magnitude")
ax[1].set_xlabel("Horizontal pixel")
ax[1].set_ylabel("Magnitude (a.u.)")

centre_row_x = np.arange(N)

# Create this line once. Later, the update function changes only its y-values.
clean_row_line, = ax[1].plot(
    centre_row_x,
    np.zeros(N),
    label="no noise, no motion",
    color="tab:blue",
    lw=2,
)

# Keep the graph axes fixed while sliders move.
ax[1].set_xlim(0, N - 1)
ax[1].set_ylim(0, 1.05)
ax[1].legend(fontsize=8)

# Add a brightness scale next to the reconstructed image.
fig.colorbar(
    clean_image,
    ax=ax[0],
    fraction=0.046,
    pad=0.04,
    label="Magnitude (a.u.)",
)

# Text below the plots showing the current TR and TE values.
status = fig.text(0.08, 0.42, "", fontsize=10)


# TR and TE controls.
# TR controls how long the tissue recovers before the next RF excitation.
# TE controls how long the transverse signal decays before measurement.
scan_specs = [
    ("TR (ms)", 100, 3000, 1200),
    ("TE (ms)", 5, 200, 80),
]

scan_sliders = []

for i, (label, low, high, initial) in enumerate(scan_specs):
    slider_axis = fig.add_axes([0.16, 0.36 - i * 0.06, 0.28, 0.020])

    scan_sliders.append(
        Slider(
            slider_axis,
            label,
            low,
            high,
            valinit=initial,
        )
    )


# T1 and T2 controls for every tissue.
# The controls use milliseconds because that is easier to read, but the code
# converts them to seconds before using them in the signal equation.
relaxation_specs = [
    ("Air T1 (ms)", 1, 6000, 1000),
    ("Air T2 (ms)", 1, 3000, 1),
    ("CSF T1 (ms)", 1, 6000, 4000),
    ("CSF T2 (ms)", 1, 3000, 1500),
    ("Fat T1 (ms)", 1, 6000, 350),
    ("Fat T2 (ms)", 1, 3000, 80),
    ("Muscle T1 (ms)", 1, 6000, 900),
    ("Muscle T2 (ms)", 1, 3000, 50),
]

relaxation_sliders = []

for i, (label, low, high, initial) in enumerate(relaxation_specs):
    slider_axis = fig.add_axes([0.66, 0.36 - i * 0.037, 0.26, 0.015])

    relaxation_sliders.append(
        Slider(
            slider_axis,
            label,
            low,
            high,
            valinit=initial,
        )
    )


# Reset button below the TR and TE sliders.
reset_button = Button(
    fig.add_axes([0.24, 0.20, 0.14, 0.04]),
    "Reset",
)


def reconstruct(kspace):
    """
    Convert k-space back into an image using the inverse 2-D Fourier transform.

    MRI scanners acquire frequency-domain data called k-space. The inverse
    Fourier transform reconstructs an image from that k-space data.
    """
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace),
            norm="ortho",
        )
    )


def update(_=None):
    """
    Recalculate the MRI signal and redraw the image and graph.

    This function runs whenever the user moves a slider.
    """

    # Read TR and TE from the sliders in milliseconds.
    tr_ms, te_ms = [slider.val for slider in scan_sliders]

    # Convert milliseconds to seconds because T1 and T2 are stored in seconds.
    tr = tr_ms * 1e-3
    te = te_ms * 1e-3


    # Read all tissue T1/T2 sliders, which are also in milliseconds.
    relaxation_ms = [slider.val for slider in relaxation_sliders]

    # Put the chosen T1 and T2 values into the correct image quadrants.
    for index, name in enumerate(("Air", "CSF", "Fat", "Muscle")):
        t1[quadrants[name]] = relaxation_ms[2 * index] * 1e-3
        t2[quadrants[name]] = relaxation_ms[2 * index + 1] * 1e-3


    # Spin-echo signal equation:
    # S = rho × (1 - exp(-TR / T1)) × exp(-TE / T2)

    # rho:
    #     Proton density. Air has rho = 0, so it produces no signal.

    # (1 - exp(-TR / T1)):
    #     T1 recovery term. Longer TR or shorter T1 produces more recovery
    #     and can increase signal.

    # exp(-TE / T2):
    #     T2 decay term. Longer TE or shorter T2 causes more decay and
    #     decreases signal.

    signal = rho * (1 - np.exp(-tr / t1)) * np.exp(-te / t2)

    # Convert the image-space signal into k-space using a 2-D Fourier transform.
    # In a real MRI scanner, gradient encoding fills k-space. This simplified
    # model uses a complete Cartesian Fourier transform instead.
    clean_kspace = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(signal),
            norm="ortho",
        )
    )

    # Reconstruct the ideal image from k-space.
    # There is no added receiver noise and no motion artifact in this model.
    clean = reconstruct(clean_kspace)

    # MRI images are normally displayed as magnitude images: |S|.
    clean_image.set_data(np.abs(clean))

    # Extract the middle horizontal row of the reconstructed image and update
    # the graph. This row passes through Fat on the left and Muscle on the right.
    row = N // 2
    clean_row_line.set_ydata(np.abs(clean[row]))

    # Update the text under the plots.
    status.set_text(
        f"TR = {tr_ms:.0f} ms     TE = {te_ms:.0f} ms"
    )

    # Request a redraw without rebuilding the figure.
    fig.canvas.draw_idle()

def reset(_):
    """Restore TR, TE, T1, and T2 controls to their starting values."""
    for slider in scan_sliders + relaxation_sliders:
        slider.reset()

# Run update() whenever any slider changes.
for slider in scan_sliders + relaxation_sliders:
    slider.on_changed(update)

# Run reset() when the Reset button is clicked.
reset_button.on_clicked(reset)

# Draw the initial image and then open the interactive window.
update()
plt.show()