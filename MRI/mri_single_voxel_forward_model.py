"""Interactive educational MRI forward model for one movable voxel.

Install dependencies once with:  pip install numpy matplotlib
Run this file with:             python mri_single_voxel_forward_model.py

This program models one finite circular voxel-region, rather than an image
made from several tissues.  Its adjustable radius makes the object large
enough to show its spatial structure in object space, k-space, and the final
reconstruction.  It still follows the MRI forward model: predict a complex
transverse signal, encode it in k-space with gradients, add optional receiver
noise, and reconstruct with an inverse Fourier transform.
"""

# Import NumPy for arrays, complex arithmetic, exponentials, and random noise.
import numpy as np
# Import Matplotlib for the plots and interactive user interface.
import matplotlib.pyplot as plt
# Import Slider and CheckButtons widgets used to change simulation parameters.
from matplotlib.widgets import Slider, CheckButtons

# Store the gyromagnetic ratio of hydrogen in cycles per second per tesla.
GAMMA_HZ = 42.577e6
# Store the nominal main magnetic field strength in tesla.
B0_T = 3.0
# Set the reconstructed field of view in metres in both in-plane directions.
FOV_M = 0.24
# Set the number of Cartesian samples in each k-space direction.
N = 64
# Set the duration of the readout gradient in seconds.
READOUT_S = 4e-3
# Set the duration of the phase-encoding gradient blip in seconds.
PHASE_S = 1e-3
# Set the rectangular RF-pulse duration in seconds.
PULSE_S = 1e-3
# Fix the proton density of the one simulated voxel-region in arbitrary units.
PROTON_DENSITY = 1.0
# Fix the voxel-region's T1 relaxation time in seconds.
T1_S = 0.900
# Fix the voxel-region's T2-star relaxation time in seconds.
T2STAR_S = 0.050
# Fix the repetition time in seconds.
TR_S = 0.500
# Fix the echo time in seconds.
TE_S = 0.020
# Fix the B1 amplitude in tesla.
B1_T = 0.10e-3
# Fix the x gradient to the Fourier-grid-matched value.
GX_TPM = N / (GAMMA_HZ * FOV_M * READOUT_S)
# Fix the y gradient to the Fourier-grid-matched value.
GY_TPM = N / (2 * GAMMA_HZ * FOV_M * PHASE_S)
# Fix the original voxel-region centre in metres.
VOXEL_X_M = -0.030
# Fix the original voxel-region centre in metres.
VOXEL_Y_M = 0.020
# Fix the radius of the one finite circular voxel-region in metres.
VOXEL_RADIUS_M = 0.025

# Make the image-space coordinate at the centre of every reconstruction pixel.
coords_m = (np.arange(N) - N / 2) * FOV_M / N
# Use rows for y and columns for x, matching a conventional displayed image.
x_grid_m, y_grid_m = np.meshgrid(coords_m, coords_m)

# Create one figure with three image panels and one diagnostic graph panel.
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
# Reserve space below the images for the diagnostic graph, controls, and status text.
plt.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.64, wspace=0.35)

# Create the first image artist for the finite reference object before any motion occurs.
object_im = axes[0].imshow(np.zeros((N, N)), cmap="gray", vmin=0, vmax=1, origin="upper")
# Label the first panel so it is not confused with the moving-object history.
axes[0].set_title("Reference object: before motion")
# Create the second image artist for logarithmic k-space magnitude.
kspace_im = axes[1].imshow(np.zeros((N, N)), cmap="magma", origin="lower", aspect="auto")
# Label the second panel; a finite object produces the familiar bright central k-space structure.
axes[1].set_title("Simulated k-space magnitude: log(1 + |S|)")
# Create the third image artist for reconstructed image magnitude.
recon_im = axes[2].imshow(np.zeros((N, N)), cmap="gray", vmin=0, vmax=1, origin="upper")
# Label the third panel.
axes[2].set_title("Inverse-FFT reconstruction")

# Add position labels to every displayed image panel.
for axis in axes:
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")

# Add colorbars so brightness has a visible numerical scale.
fig.colorbar(object_im, ax=axes[0], fraction=0.046, pad=0.04, label="magnitude (a.u.)")
# Add the k-space magnitude colorbar.
fig.colorbar(kspace_im, ax=axes[1], fraction=0.046, pad=0.04, label="log(1 + magnitude)")
# Add the reconstruction colorbar.
fig.colorbar(recon_im, ax=axes[2], fraction=0.046, pad=0.04, label="magnitude (a.u.)")

# Add an axes for the difference between moving and stationary k-space samples.
signal_axis = fig.add_axes([0.08, 0.39, 0.56, 0.16])
# Add an axes used only for a concise state and warning message.
status_text = fig.text(0.08, 0.345, "", fontsize=9)

# Define slider label, minimum, maximum, and initial value for the remaining controls.
slider_specs = [
    ("Motion dx (mm)", -80.0, 80.0, 25.0),
    ("Motion dy (mm)", -80.0, 80.0, -20.0),
    ("B0 inhomogeneity (ppm/FOV)", 0.0, 5.0, 0.0),
    ("Thermal noise SD", 0.0, 0.02, 0.00),
    ("Electrical noise SD", 0.0, 0.02, 0.00),
    ("Noise seed", 0.0, 10000.0, 123.0),
]

# Keep every Slider object in a list so callbacks and values are easy to access.
sliders = []
# Create one horizontal slider for each requested parameter.
for index, (label, low, high, initial) in enumerate(slider_specs):
    # Place each slider in its own thin axes beneath the status line.
    slider_axis = fig.add_axes([0.29, 0.30 - index * 0.030, 0.47, 0.016])
    # Construct and store the interactive slider.
    sliders.append(Slider(slider_axis, label, low, high, valinit=initial))

# Create a small checkbox panel for enabling the optional effects.
check_axis = fig.add_axes([0.80, 0.11, 0.17, 0.18])
# Motion is enabled initially so its consequence is immediately visible.
checks = CheckButtons(check_axis, ["Enable motion", "Thermal noise", "Electrical noise"], [True, False, False])
# Put a concise title above the checkbox panel.
check_axis.set_title("Optional effects", fontsize=9)


def spoiled_gre_signal(rho, t1_s, t2star_s, tr_s, te_s, b1_t):
    """Calculate the complex-free magnitude at TE for a spoiled GRE voxel."""
    # Convert B1 amplitude and pulse duration into flip angle in radians.
    alpha_rad = 2 * np.pi * GAMMA_HZ * b1_t * PULSE_S
    # Calculate the longitudinal recovery factor over one repetition period.
    e1 = np.exp(-tr_s / t1_s)
    # Calculate the spoiled-GRE steady-state transverse signal just after RF.
    steady_transverse = rho * (1 - e1) * np.sin(alpha_rad) / (1 - e1 * np.cos(alpha_rad))
    # Apply T2-star loss during the time between RF excitation and the echo.
    return steady_transverse * np.exp(-te_s / t2star_s), alpha_rad


def make_circular_voxel(signal_at_te, x_m, y_m, radius_m):
    """Make one finite, uniformly bright circular voxel-region at a physical location."""
    # Calculate every image pixel's squared distance from the circle centre.
    squared_distance_m2 = (x_grid_m - x_m) ** 2 + (y_grid_m - y_m) ** 2
    # Mark pixels inside the requested radius as belonging to the finite voxel-region.
    inside_voxel = squared_distance_m2 <= radius_m ** 2
    # Assign the predicted TE signal inside the region and zero signal outside it.
    return signal_at_te * inside_voxel.astype(float)


def apply_b0_inhomogeneity(object_image, inhomogeneity_ppm_per_fov):
    """Apply the TE phase caused by a linear static-B0 error across one FOV."""
    # Convert the requested parts-per-million variation across the FOV into a local field error in tesla.
    delta_b0_t = B0_T * inhomogeneity_ppm_per_fov * 1e-6 * (x_grid_m / FOV_M)
    # Convert that local field error into off-resonance phase accumulated by the echo time.
    off_resonance_phase = -2 * np.pi * GAMMA_HZ * delta_b0_t * TE_S
    # Return the complex object after its spatially varying B0 phase has been applied.
    return object_image * np.exp(1j * off_resonance_phase)


def make_kspace(object_before, object_after, gx_tpm, gy_tpm, motion_on, thermal_on, electrical_on, thermal_sd, electrical_sd, seed):
    """Apply the Fourier forward model to the finite voxel before and after a motion event."""
    # Create the readout kx locations in cycles per metre.
    kx = GAMMA_HZ * gx_tpm * np.linspace(-READOUT_S / 2, READOUT_S / 2, N, endpoint=False)
    # Create the phase-encode ky locations in cycles per metre.
    ky = GAMMA_HZ * gy_tpm * PHASE_S * np.linspace(-1, 1, N, endpoint=False)
    # Make the x-direction Fourier phase matrix for every kx location and image x coordinate.
    ex = np.exp(-2j * np.pi * np.outer(kx, coords_m))
    # Make the y-direction Fourier phase matrix for every ky location and image y coordinate.
    ey = np.exp(-2j * np.pi * np.outer(ky, coords_m))
    # Transform the reference object into its complete stationary k-space data matrix.
    kspace_before = ey @ object_before @ ex.T / N
    # Transform the displaced object into the k-space data that would be acquired after movement.
    kspace_after = ey @ object_after @ ex.T / N
    # Start with the data from the object before it moves.
    kspace = kspace_before.copy()
    # Place the instantaneous movement halfway through the ordered acquisition.
    motion_sample = (N * N) // 2
    # Replace only the second half of acquired samples with data from the displaced object.
    if motion_on:
        # This is the data inconsistency that creates motion artifact after reconstruction.
        kspace.flat[motion_sample:] = kspace_after.flat[motion_sample:]
    # Make a repeatable random-number generator so the seed selects a noise realization.
    rng = np.random.default_rng(seed)
    # Add circular complex Gaussian thermal noise if the thermal checkbox is enabled.
    if thermal_on and thermal_sd > 0:
        # Divide by sqrt(2) so thermal_sd is the complex noise RMS convention used here.
        kspace += thermal_sd * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2)
    # Add independent circular complex Gaussian receiver-electronics noise if requested.
    if electrical_on and electrical_sd > 0:
        # This has the same mathematical distribution as thermal receiver noise, but a separate source level.
        kspace += electrical_sd * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2)
    # Return acquired k-space, stationary k-space, and physical sample-coordinate vectors for diagnostics.
    return kspace, kspace_before, kx, ky


def update(_=None):
    """Read controls, run the MRI forward model, and redraw every plot."""
    # Convert the prescribed motion in x from millimetres to metres.
    dx_m = sliders[0].val * 1e-3
    # Convert the prescribed motion in y from millimetres to metres.
    dy_m = sliders[1].val * 1e-3
    # Read the linear B0 variation across the field of view in parts per million.
    inhomogeneity_ppm_per_fov = sliders[2].val
    # Read the thermal-noise standard deviation.
    thermal_sd = sliders[3].val
    # Read the electronics-noise standard deviation.
    electrical_sd = sliders[4].val
    # Round the final slider to obtain an integer random seed.
    seed = int(round(sliders[5].val))
    # Read the three checkbox states in their displayed order.
    motion_on, thermal_on, electrical_on = checks.get_status()

    # Predict the fixed finite voxel-region's magnitude at TE and calculate its RF flip angle.
    signal_at_te, alpha_rad = spoiled_gre_signal(PROTON_DENSITY, T1_S, T2STAR_S, TR_S, TE_S, B1_T)
    # Make the finite object at its initial position before the motion event.
    object_before = make_circular_voxel(signal_at_te, VOXEL_X_M, VOXEL_Y_M, VOXEL_RADIUS_M)
    # Make the same finite object at its displaced position after the motion event.
    object_after = make_circular_voxel(signal_at_te, VOXEL_X_M + dx_m, VOXEL_Y_M + dy_m, VOXEL_RADIUS_M)
    # Add static spatially varying B0 phase to the object before movement.
    object_before = apply_b0_inhomogeneity(object_before, inhomogeneity_ppm_per_fov)
    # Add the same scanner B0 phase map to the object after movement.
    object_after = apply_b0_inhomogeneity(object_after, inhomogeneity_ppm_per_fov)
    # Generate the acquired k-space from the forward model and add any enabled noise.
    kspace, stationary_kspace, kx, ky = make_kspace(object_before, object_after, GX_TPM, GY_TPM, motion_on, thermal_on, electrical_on, thermal_sd, electrical_sd, seed)
    # Invert the sampled k-space with the standard unitary two-dimensional inverse DFT.
    reconstruction = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(kspace), norm="ortho"))

    # Display the initial finite object's magnitude, because B0 inhomogeneity makes its stored signal complex.
    object_display = np.abs(object_before)

    # Update the object panel with the current point locations.
    object_im.set_data(object_display)
    # Update the object brightness scale without dividing by zero for a zero-density voxel.
    object_im.set_clim(0, max(signal_at_te, 1e-9))
    # Update the k-space panel with logarithmic magnitude so its central low-frequency energy is visible.
    kspace_im.set_data(np.log1p(np.abs(kspace)))
    # Scale the k-space display to its current peak.
    kspace_im.set_clim(0, max(np.log1p(np.abs(kspace)).max(), 1e-9))
    # Update the reconstructed-magnitude panel.
    recon_im.set_data(np.abs(reconstruction))
    # Update the reconstruction brightness scale.
    recon_im.set_clim(0, max(np.abs(reconstruction).max(), 1e-9))

    # Calculate the magnitude of data inconsistency relative to a stationary, noiseless scan.
    kspace_difference = np.abs(kspace - stationary_kspace)
    # Clear the previous k-space-difference trace before drawing the new one.
    signal_axis.clear()
    # Plot difference in exact acquisition order; motion creates nonzero data only after the event.
    signal_axis.plot(kspace_difference.ravel(), color="tab:blue", linewidth=1)
    # Mark the exact sample at which motion begins when enabled.
    if motion_on:
        # Draw the boundary between data acquired before and after the displacement.
        signal_axis.axvline(N * N // 2, color="tab:red", linestyle="--", label="motion event")
    # Label the diagnostic plot with the comparison it makes.
    signal_axis.set_title("K-space difference from a stationary finite voxel")
    # Label its horizontal axis.
    signal_axis.set_xlabel("acquisition sample number")
    # Label its vertical axis.
    signal_axis.set_ylabel("|acquired − stationary| (a.u.)")
    # Show the motion-event legend only if the event exists.
    if motion_on:
        signal_axis.legend(fontsize=8)

    # Check whether the selected gradients still use the nominal discrete Fourier sample spacing.
    kx_spacing = abs(kx[1] - kx[0])
    # Compute the corresponding y-direction k-space spacing.
    ky_spacing = abs(ky[1] - ky[0])
    # Declare the grid matched only when both spacings are near one over the fixed FOV.
    grid_matched = np.isclose(kx_spacing, 1 / FOV_M, rtol=0.01) and np.isclose(ky_spacing, 1 / FOV_M, rtol=0.01)
    # Check a minimal simplified sequence-timing condition.
    timing_ok = TE_S >= PULSE_S / 2 + PHASE_S + READOUT_S / 2 and TR_S >= TE_S + READOUT_S / 2
    # State what is being simulated and warn about nonphysical timing or grid mismatch.
    status_text.set_text(f"Fixed TE signal: {signal_at_te:.4f} a.u.   |   B0: {B0_T:.1f} T   |   "
                         f"B0 variation: {inhomogeneity_ppm_per_fov:.2f} ppm/FOV   |   motion: {'on' if motion_on else 'off'}   |   "
                         f"Fourier grid: {'matched' if grid_matched else 'mismatched'}   |   timing: {'feasible' if timing_ok else 'not feasible'}")
    # Use red text whenever either important diagnostic check fails.
    status_text.set_color("black" if grid_matched and timing_ok else "tab:red")
    # Ask Matplotlib to redraw without blocking interaction.
    fig.canvas.draw_idle()


# Re-run the model whenever any slider value changes.
for slider in sliders:
    # Register the shared redraw callback for this slider.
    slider.on_changed(update)

# Re-run the model whenever any optional-effect checkbox changes.
checks.on_clicked(update)
# Draw the first complete simulation before the window is shown.
update()
# Start Matplotlib's interactive window loop.
plt.show()
