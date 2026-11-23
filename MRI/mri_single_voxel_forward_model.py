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
# Import colour-map helpers for the fixed-scale three-dimensional B0 surface.
from matplotlib import cm
# Import a fixed normalization so B0 colours do not rescale between updates.
from matplotlib.colors import Normalize
# Import Slider, CheckButtons, and Button widgets used to change simulation parameters.
from matplotlib.widgets import Slider, CheckButtons, Button

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
# Fix the random-number seed so receiver-noise realizations remain reproducible without a UI control.
NOISE_SEED = 123
# Fix the B0-map display range in ppm so colour changes are directly comparable between settings.
B0_DISPLAY_LIMIT_PPM = 35.0

# Make the image-space coordinate at the centre of every reconstruction pixel.
coords_m = (np.arange(N) - N / 2) * FOV_M / N
# Use rows for y and columns for x, matching a conventional displayed image.
x_grid_m, y_grid_m = np.meshgrid(coords_m, coords_m)

# Create one taller window containing the images, diagnostic plots, sliders, and effect options.
fig, axes = plt.subplots(1, 3, figsize=(14, 10))
# Reserve a generous gap between the upper images and the lower diagnostic panels.
plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.73, wspace=0.35)

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
recon_colourbar = fig.colorbar(recon_im, ax=axes[2], fraction=0.046, pad=0.04, label="magnitude (a.u.)")
# Read the actual displayed bounds after image aspect ratios and colourbars have been applied.
reference_bounds = axes[0].get_position()
# Read the k-space panel bounds that define the intended centre spacing.
kspace_bounds = axes[1].get_position()
# Read the reconstruction panel's current displayed bounds before moving only that panel.
reconstruction_bounds = axes[2].get_position()
# Calculate the reference and k-space horizontal centres in normalized figure coordinates.
reference_centre = reference_bounds.x0 + reference_bounds.width / 2
# Calculate the k-space horizontal centre in normalized figure coordinates.
kspace_centre = kspace_bounds.x0 + kspace_bounds.width / 2
# Place the reconstruction centre one equal panel-to-panel interval to the right of k-space.
target_reconstruction_centre = 2 * kspace_centre - reference_centre
# Convert the desired centre back into the reconstruction axis's left position.
target_reconstruction_left = target_reconstruction_centre - reconstruction_bounds.width / 2
# Move only the reconstruction image panel while retaining its existing size and vertical placement.
axes[2].set_position([target_reconstruction_left, reconstruction_bounds.y0, reconstruction_bounds.width, reconstruction_bounds.height])
# Place the reconstruction colourbar directly beside the newly positioned reconstruction panel.
recon_colourbar.ax.set_position([target_reconstruction_left + reconstruction_bounds.width + 0.010, reconstruction_bounds.y0, recon_colourbar.ax.get_position().width, reconstruction_bounds.height])

# Add a taller three-dimensional axes so its square 3-D drawing area is large enough to read.
b0_axis = fig.add_axes([0.06, 0.39, 0.42, 0.22], projection="3d")
# Create one fixed colour normalization shared by every B0 surface redraw.
b0_norm = Normalize(vmin=-B0_DISPLAY_LIMIT_PPM, vmax=B0_DISPLAY_LIMIT_PPM)
# Create the initial flat B0 surface before any field modifier is enabled.
b0_surface = b0_axis.plot_surface(x_grid_m * 1e3, y_grid_m * 1e3, np.zeros((N, N)), cmap="coolwarm", norm=b0_norm, linewidth=0, antialiased=False)
# Label the B0-surface panel with a compact title that stays inside its dedicated area.
b0_axis.set_title("Configured B0 offset surface", fontsize=12, pad=4)
# Label its fixed physical x direction.
b0_axis.set_xlabel("x (mm)", fontsize=10, labelpad=2)
# Label its fixed physical y direction.
b0_axis.set_ylabel("y (mm)", fontsize=10, labelpad=2)
# Omit a z-axis label because the adjacent colourbar already supplies the same units without crowding the plot.
# Lock the x limits to the full field of view in millimetres.
b0_axis.set_xlim(-FOV_M * 500, FOV_M * 500)
# Lock the y limits to the full field of view in millimetres.
b0_axis.set_ylim(-FOV_M * 500, FOV_M * 500)
# Lock the vertical limits to the same range used by the colour map.
b0_axis.set_zlim(-B0_DISPLAY_LIMIT_PPM, B0_DISPLAY_LIMIT_PPM)
# Choose a stable viewing angle that shows both field shape and location.
b0_axis.view_init(elev=28, azim=-55)
# Create a colourbar from a fixed scalar mapper rather than from a changing surface artist.
b0_colour_mapper = cm.ScalarMappable(norm=b0_norm, cmap="coolwarm")
# Attach an empty array so Matplotlib can use the scalar mapper for the colourbar.
b0_colour_mapper.set_array([])
# Add the fixed B0 heat-map scale to the left of the three-dimensional surface.
b0_colourbar = fig.colorbar(b0_colour_mapper, ax=b0_axis, location="left", fraction=0.046, pad=0.04, label="B0 offset (ppm)")
# Reduce colourbar text slightly so it does not crowd the neighbouring diagnostic graph.
b0_colourbar.ax.tick_params(labelsize=9)
# Reduce the colourbar label size to match the compact B0 panel.
b0_colourbar.set_label("B0 offset (ppm)", fontsize=10)
# Add an axes for the difference between acquired and ideal k-space samples.
signal_axis = fig.add_axes([0.58, 0.42, 0.38, 0.16])
# Add an axes used only for a concise state and warning message.
status_text = fig.text(0.08, 0.325, "", fontsize=9)

# Define slider label, minimum, maximum, and initial value for the remaining controls.
slider_specs = [
    ("Motion dx (mm)", -80.0, 80.0, 25.0),
    ("Motion dy (mm)", -80.0, 80.0, -20.0),
    ("Uniform B0 floor (ppm)", 0.0, 10.0, 0.0),
    ("Linear B0 ramp (ppm/FOV)", -10.0, 10.0, 0.0),
    ("Local B0 bump (ppm)", 0.0, 20.0, 0.0),
    ("Bump x position (mm)", -100.0, 100.0, 0.0),
    ("Bump y position (mm)", -100.0, 100.0, 0.0),
    ("Bump radius (mm)", 5.0, 90.0, 35.0),
    ("Receiver noise SD", 0.0, 0.75, 0.00),
]

# Keep every Slider object in a list so callbacks and values are easy to access.
sliders = []
# Create one horizontal slider for each requested parameter.
for index, (label, low, high, initial) in enumerate(slider_specs):
    # Place each slider in its own thin axes beneath the status line.
    slider_axis = fig.add_axes([0.27, 0.275 - index * 0.023, 0.36, 0.015])
    # Construct and store the interactive slider.
    sliders.append(Slider(slider_axis, label, low, high, valinit=initial))

# Create a checkbox panel for turning each optional physical effect on or off.
check_axis = fig.add_axes([0.75, 0.055, 0.21, 0.190])
# Store the desired startup state for every checkbox so the reset button can restore it exactly.
INITIAL_CHECK_STATES = [True, False, False, False, False]
# Motion is enabled initially, while B0 modifiers and noise begin disabled.
checks = CheckButtons(check_axis, ["Enable motion", "Receiver noise", "Uniform B0 floor", "Linear B0 ramp", "Local B0 bump"], INITIAL_CHECK_STATES)
# Put a concise title above the checkbox panel.
check_axis.set_title("Optional effects", fontsize=9)
# Add a reset button directly above the options box.
reset_axis = fig.add_axes([0.84, 0.295, 0.12, 0.020])
# Create the button that restores all user controls to their initial values.
reset_button = Button(reset_axis, "Reset Button")


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


def make_b0_off_resonance_map(floor_ppm, ramp_ppm_per_fov, bump_ppm, bump_x_m, bump_y_m, bump_radius_m):
    """Make the scanner B0 map from a uniform floor, linear ramp, and movable curved local bump."""
    # Make the uniform positive field offset that raises the B0 value everywhere by the same amount.
    uniform_component_ppm = floor_ppm
    # Make a smooth linear weak-to-strong field ramp along x; negative values reverse its direction.
    linear_component_ppm = ramp_ppm_per_fov * (x_grid_m / FOV_M)
    # Calculate squared distance from the centre of the selected local B0 imperfection.
    bump_distance_squared_m2 = (x_grid_m - bump_x_m) ** 2 + (y_grid_m - bump_y_m) ** 2
    # Make a Gaussian-shaped local B0 bump whose radius controls its spatial spread.
    bump_component_ppm = bump_ppm * np.exp(-bump_distance_squared_m2 / (2 * bump_radius_m ** 2))
    # Add all three user-controlled contributions to make the total B0 offset map in parts per million.
    total_offset_ppm = uniform_component_ppm + linear_component_ppm + bump_component_ppm
    # Convert the total field offset from ppm into tesla relative to the nominal B0 value.
    delta_b0_t = B0_T * total_offset_ppm * 1e-6
    # Convert the local B0 error to an off-resonance frequency map in cycles per second (hertz).
    off_resonance_hz = GAMMA_HZ * delta_b0_t
    # Return both the frequency map used in encoding and the ppm map used for display.
    return off_resonance_hz, total_offset_ppm


def forward_encode(object_image, kx, ky, readout_time_s, off_resonance_hz):
    """Encode one object while B0 phase continues to accumulate throughout the readout."""
    # Make the x-direction Fourier phase matrix for every kx location and image x coordinate.
    ex = np.exp(-2j * np.pi * np.outer(kx, coords_m))
    # Make the y-direction Fourier phase matrix for every ky location and image y coordinate.
    ey = np.exp(-2j * np.pi * np.outer(ky, coords_m))
    # Allocate one complex value for every phase-encode and readout sample.
    encoded_kspace = np.empty((N, N), dtype=complex)
    # Calculate every readout sample separately because off-resonance phase changes with time.
    for column, time_from_te_s in enumerate(readout_time_s):
        # Accumulate the local B0 phase from excitation through this particular sample time.
        b0_phase = np.exp(-2j * np.pi * off_resonance_hz * (TE_S + time_from_te_s))
        # Apply that time-dependent phase to the object before spatial encoding this sample.
        object_at_sample = object_image * b0_phase
        # Sum the Fourier-encoded contribution of every object pixel for every ky value at this kx value.
        encoded_kspace[:, column] = ey @ object_at_sample @ ex[column, :] / N
    # Return the fully encoded k-space data matrix.
    return encoded_kspace


def make_kspace(object_before, object_after, gx_tpm, gy_tpm, off_resonance_hz, motion_on, noise_on, noise_sd, seed):
    """Apply the Fourier forward model to the finite voxel before and after a motion event."""
    # Create the readout kx locations in cycles per metre.
    kx = GAMMA_HZ * gx_tpm * np.linspace(-READOUT_S / 2, READOUT_S / 2, N, endpoint=False)
    # Create the phase-encode ky locations in cycles per metre.
    ky = GAMMA_HZ * gy_tpm * PHASE_S * np.linspace(-1, 1, N, endpoint=False)
    # Make one timing value for every readout sample, with the k-space centre acquired at TE.
    readout_time_s = np.linspace(-READOUT_S / 2, READOUT_S / 2, N, endpoint=False)
    # Transform the reference object into its complete stationary k-space data matrix.
    kspace_before = forward_encode(object_before, kx, ky, readout_time_s, off_resonance_hz)
    # Transform the displaced object into the k-space data that would be acquired after movement.
    kspace_after = forward_encode(object_after, kx, ky, readout_time_s, off_resonance_hz)
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
    # Add circular complex Gaussian receiver noise if the receiver-noise checkbox is enabled.
    if noise_on and noise_sd > 0:
        # Divide by sqrt(2) so noise_sd is the requested complex-noise RMS magnitude.
        kspace += noise_sd * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2)
    # Return acquired k-space, stationary k-space, and physical sample-coordinate vectors for diagnostics.
    return kspace, kspace_before, kx, ky


def update(_=None):
    """Read controls, run the MRI forward model, and redraw every plot."""
    # Allow this function to replace the existing three-dimensional B0 surface artist.
    global b0_surface
    # Convert the prescribed motion in x from millimetres to metres.
    dx_m = sliders[0].val * 1e-3
    # Convert the prescribed motion in y from millimetres to metres.
    dy_m = sliders[1].val * 1e-3
    # Read the uniform B0 floor offset in parts per million.
    floor_ppm = sliders[2].val
    # Read the signed left-to-right B0 ramp strength in parts per million per field of view.
    ramp_ppm_per_fov = sliders[3].val
    # Read the peak size of the movable local B0 bump in parts per million.
    bump_ppm = sliders[4].val
    # Convert the local B0 bump's x position from millimetres to metres.
    bump_x_m = sliders[5].val * 1e-3
    # Convert the local B0 bump's y position from millimetres to metres.
    bump_y_m = sliders[6].val * 1e-3
    # Convert the local B0 bump's radius from millimetres to metres.
    bump_radius_m = sliders[7].val * 1e-3
    # Read the combined receiver-noise standard deviation.
    noise_sd = sliders[8].val
    # Use the fixed seed so a control change does not also create an unrelated new noise realization.
    seed = NOISE_SEED
    # Read the five checkbox states in their displayed order.
    motion_on, noise_on, floor_on, ramp_on, bump_on = checks.get_status()
    # Use the floor slider only while its modifier is enabled.
    active_floor_ppm = floor_ppm if floor_on else 0.0
    # Use the ramp slider only while its modifier is enabled.
    active_ramp_ppm_per_fov = ramp_ppm_per_fov if ramp_on else 0.0
    # Use the bump parameters only while the local-bump modifier is enabled.
    active_bump_ppm = bump_ppm if bump_on else 0.0

    # Predict the fixed finite voxel-region's magnitude at TE and calculate its RF flip angle.
    signal_at_te, alpha_rad = spoiled_gre_signal(PROTON_DENSITY, T1_S, T2STAR_S, TR_S, TE_S, B1_T)
    # Make the finite object at its initial position before the motion event.
    object_before = make_circular_voxel(signal_at_te, VOXEL_X_M, VOXEL_Y_M, VOXEL_RADIUS_M)
    # Make the same finite object at its displaced position after the motion event.
    object_after = make_circular_voxel(signal_at_te, VOXEL_X_M + dx_m, VOXEL_Y_M + dy_m, VOXEL_RADIUS_M)
    # Make the scanner-fixed B0 map that causes spatially varying off-resonance frequency.
    off_resonance_hz, b0_offset_ppm = make_b0_off_resonance_map(active_floor_ppm, active_ramp_ppm_per_fov, active_bump_ppm, bump_x_m, bump_y_m, bump_radius_m)
    # Generate the acquired k-space from the forward model and add any enabled noise.
    kspace, stationary_kspace, kx, ky = make_kspace(object_before, object_after, GX_TPM, GY_TPM, off_resonance_hz, motion_on, noise_on, noise_sd, seed)
    # Make the ideal reference scan: stationary object, perfectly homogeneous B0, and no receiver noise.
    ideal_kspace, _, _, _ = make_kspace(object_before, object_before, GX_TPM, GY_TPM, np.zeros_like(off_resonance_hz), False, False, 0.0, seed)
    # Invert the sampled k-space with the standard unitary two-dimensional inverse DFT.
    reconstruction = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(kspace), norm="ortho"))

    # Display the initial finite object's magnitude; B0 phase is invisible in an object-magnitude view.
    object_display = object_before

    # Update the object panel with the fixed finite reference object before its motion event.
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
    # Remove the previous B0 surface before drawing the new configured field surface.
    b0_surface.remove()
    # Draw the updated B0 surface using the fixed colour normalization and physical coordinate grid.
    b0_surface = b0_axis.plot_surface(x_grid_m * 1e3, y_grid_m * 1e3, b0_offset_ppm, cmap="coolwarm", norm=b0_norm, linewidth=0, antialiased=False)
    # Reassert fixed x limits because the plotted surface must never change the displayed spatial extent.
    b0_axis.set_xlim(-FOV_M * 500, FOV_M * 500)
    # Reassert fixed y limits because the plotted surface must never change the displayed spatial extent.
    b0_axis.set_ylim(-FOV_M * 500, FOV_M * 500)
    # Reassert fixed z limits because field-height changes must remain directly comparable.
    b0_axis.set_zlim(-B0_DISPLAY_LIMIT_PPM, B0_DISPLAY_LIMIT_PPM)

    # Calculate the total k-space change relative to an ideal stationary scan with homogeneous B0.
    kspace_difference = np.abs(kspace - ideal_kspace)
    # Clear the previous k-space-difference trace before drawing the new one.
    signal_axis.clear()
    # Plot difference in exact acquisition order; B0, motion, and noise can each make it nonzero.
    signal_axis.plot(kspace_difference.ravel(), color="tab:blue", linewidth=1)
    # Mark the exact sample at which motion begins when enabled.
    if motion_on:
        # Draw the boundary between data acquired before and after the displacement.
        signal_axis.axvline(N * N // 2, color="tab:red", linestyle="--", label="motion event")
    # Label the diagnostic plot with the comparison it makes.
    signal_axis.set_title("K-space difference from ideal stationary homogeneous-B0 scan")
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
                         f"floor: {active_floor_ppm:.2f} ppm   |   ramp: {active_ramp_ppm_per_fov:.2f} ppm/FOV   |   bump: {active_bump_ppm:.2f} ppm   |   "
                         f"motion: {'on' if motion_on else 'off'}   |   "
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


def reset_controls(_):
    """Restore all sliders and checkboxes to their original simulation settings."""
    # Reset every slider to the valinit value declared in slider_specs.
    for slider in sliders:
        slider.reset()
    # Read the currently active checkbox states before changing any of them.
    current_states = checks.get_status()
    # Toggle only controls whose current state differs from the desired startup state.
    for index, desired_state in enumerate(INITIAL_CHECK_STATES):
        if current_states[index] != desired_state:
            checks.set_active(index)
    # Draw once after all controls have been restored.
    update()


# Connect the reset-button click to the full control-reset routine.
reset_button.on_clicked(reset_controls)
# Draw the first complete simulation before the window is shown.
update()
# Start Matplotlib's interactive window loop.
plt.show()
