#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 08:20:16 2026

@author: ashlyn.nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# Illustrative relaxivities in s^-1 mM^-1.
# Gadolinium shortens both T1 and T2:
# R1 = 1/T1 = R1_baseline + r1 * [Gd]
# R2 = 1/T2 = R2_baseline + r2 * [Gd]
R1_RELAXIVITY = 4.5
R2_RELAXIVITY = 5.0

DEFAULT_TR_MS = 600.0
DEFAULT_TE_MS = 15.0


def spin_echo_signal(t1_s, t2_s, tr_ms, te_ms):
    """Relative spin-echo signal."""
    tr_s = tr_ms / 1000
    te_s = te_ms / 1000
    return (1 - np.exp(-tr_s / t1_s)) * np.exp(-te_s / t2_s)


fig, axes = plt.subplots(1, 3, figsize=(15, 6))
plt.subplots_adjust(left=0.08, right=0.97, bottom=0.40, wspace=0.32)

ax_t1, ax_t2, ax_signal = axes

recovery_ms = np.linspace(0, 3000, 500)
echo_ms = np.linspace(0, 300, 500)
dose_points = np.linspace(0, 10, 300)

# T1 recovery graph
t1_before_line, = ax_t1.plot([], [], "--", color="gray", label="Before Gd")
t1_after_line, = ax_t1.plot([], [], color="firebrick", lw=2.5, label="With Gd")
t1_tr_marker = ax_t1.axvline(
    DEFAULT_TR_MS, color="black", lw=1, ls=":", label="TR"
)

ax_t1.set(
    title="T1 longitudinal recovery",
    xlabel="Time after excitation (ms)",
    ylabel="Longitudinal magnetization (relative)",
    xlim=(0, 3000),
    ylim=(0, 1.04),
)
ax_t1.legend(loc="lower right")

# T2 decay graph
t2_before_line, = ax_t2.plot([], [], "--", color="gray", label="Before Gd")
t2_after_line, = ax_t2.plot([], [], color="steelblue", lw=2.5, label="With Gd")
t2_te_marker = ax_t2.axvline(
    DEFAULT_TE_MS, color="black", lw=1, ls=":", label="TE"
)

ax_t2.set(
    title="T2 transverse decay",
    xlabel="Echo time (ms)",
    ylabel="Transverse magnetization (relative)",
    xlim=(0, 300),
    ylim=(0, 1.04),
)
ax_t2.legend(loc="upper right")

# Signal-versus-gadolinium graph
signal_line, = ax_signal.plot([], [], color="forestgreen", lw=2.5)
marker, = ax_signal.plot([], [], "o", color="darkorange", ms=9)

ax_signal.set(
    title="Spin-echo signal",
    xlabel="Local gadolinium concentration (mM)",
    ylabel="Relative signal",
    xlim=(0, 10),
    ylim=(0, 1.05),
)

info = ax_signal.text(
    0.03,
    0.97,
    "",
    transform=ax_signal.transAxes,
    va="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)

# Sliders
dose_slider = Slider(
    fig.add_axes([0.18, 0.30, 0.67, 0.025]),
    "Local Gd concentration (mM)",
    0,
    10,
    valinit=1.0,
    valstep=0.05,
)

t1_slider = Slider(
    fig.add_axes([0.18, 0.245, 0.67, 0.025]),
    "Baseline T1 (ms)",
    100,
    3000,
    valinit=1000,
    valstep=10,
)

t2_slider = Slider(
    fig.add_axes([0.18, 0.19, 0.67, 0.025]),
    "Baseline T2 (ms)",
    10,
    500,
    valinit=100,
    valstep=5,
)

tr_slider = Slider(
    fig.add_axes([0.18, 0.135, 0.67, 0.025]),
    "TR (ms)",
    50,
    3000,
    valinit=DEFAULT_TR_MS,
    valstep=10,
)

te_slider = Slider(
    fig.add_axes([0.18, 0.08, 0.67, 0.025]),
    "TE (ms)",
    1,
    300,
    valinit=DEFAULT_TE_MS,
    valstep=1,
)

reset_button = Button(
    fig.add_axes([0.43, 0.015, 0.14, 0.04]),
    "Reset",
)


def update(_=None):
    concentration = dose_slider.val
    baseline_t1_s = t1_slider.val / 1000
    baseline_t2_s = t2_slider.val / 1000
    tr_ms = tr_slider.val
    te_ms = te_slider.val

    # Calculate effective T1 and T2 after gadolinium.
    baseline_r1 = 1 / baseline_t1_s
    baseline_r2 = 1 / baseline_t2_s

    t1_after_s = 1 / (baseline_r1 + R1_RELAXIVITY * concentration)
    t2_after_s = 1 / (baseline_r2 + R2_RELAXIVITY * concentration)

    # Update T1 recovery curves.
    t1_before = 1 - np.exp(-(recovery_ms / 1000) / baseline_t1_s)
    t1_after = 1 - np.exp(-(recovery_ms / 1000) / t1_after_s)

    t1_before_line.set_data(recovery_ms, t1_before)
    t1_after_line.set_data(recovery_ms, t1_after)
    t1_tr_marker.set_xdata([tr_ms, tr_ms])

    # Update T2 decay curves.
    t2_before = np.exp(-(echo_ms / 1000) / baseline_t2_s)
    t2_after = np.exp(-(echo_ms / 1000) / t2_after_s)

    t2_before_line.set_data(echo_ms, t2_before)
    t2_after_line.set_data(echo_ms, t2_after)
    t2_te_marker.set_xdata([te_ms, te_ms])

    # Update spin-echo signal-versus-dose curve.
    t1_curve_s = 1 / (baseline_r1 + R1_RELAXIVITY * dose_points)
    t2_curve_s = 1 / (baseline_r2 + R2_RELAXIVITY * dose_points)

    signal_curve = spin_echo_signal(t1_curve_s, t2_curve_s, tr_ms, te_ms)
    current_signal = spin_echo_signal(t1_after_s, t2_after_s, tr_ms, te_ms)

    signal_line.set_data(dose_points, signal_curve)
    marker.set_data([concentration], [current_signal])

    ax_signal.set_title(
        f"Spin-echo signal (TR={tr_ms:.0f} ms, TE={te_ms:.0f} ms)"
    )

    info.set_text(
        f"Effective T1: {t1_after_s * 1000:.0f} ms\n"
        f"Effective T2: {t2_after_s * 1000:.1f} ms\n"
        f"Relative signal: {current_signal:.3f}"
    )

    fig.canvas.draw_idle()


def reset(_):
    dose_slider.reset()
    t1_slider.reset()
    t2_slider.reset()
    tr_slider.reset()
    te_slider.reset()


for slider in (dose_slider, t1_slider, t2_slider, tr_slider, te_slider):
    slider.on_changed(update)

reset_button.on_clicked(reset)

update()
plt.show()