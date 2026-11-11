"""Explore a simple MRI forward model with four tissue types.

Install: pip install numpy matplotlib
Run:     python "mri_quadrants(1).py"

Move the sliders to change the scan. The three images show the tissue signal,
the measured spatial-frequency data (k-space), and the reconstructed image.
The graphs show the echo and recovery between RF pulses.

This is an educational approximation of a spoiled gradient-echo scan, not a
full simulation of all MRI physics. Intensities use arbitrary units (a.u.).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Physical constants and fixed acquisition settings. All calculations use SI
# units: seconds, metres, tesla and hertz. Sliders display more convenient units.
N = 32                     # Number of pixels in each image direction
FOV = 0.24                 # Field of view: width and height of the image (m)
GAMMA_HZ = 42.577e6        # Hydrogen frequency change per tesla (Hz/T)
PULSE_S = 1e-3             # Duration of the rectangular RF pulse (s)
READOUT_S = 4e-3
PHASE_S = 1e-3
# Centre the pixel coordinates around zero, as in a centred Fourier transform.
coords = (np.arange(N) - N // 2) * FOV / N
x, _ = np.meshgrid(coords, coords)

# Illustrative values at roughly 1.5 T: proton density, T1 (s), T2* (s).
tissues = {"Air": (0.0, 1.0, 0.001), "CSF": (1.0, 4.0, 0.50),
           "Fat": (0.9, 0.35, 0.060), "Muscle": (0.8, 0.90, 0.035)}
# rho is relative proton density. T1 describes longitudinal recovery; T2*
# describes loss of transverse signal. The four arrays represent image pixels.
rho = np.zeros((N, N))
t1 = np.empty((N, N))
t2s = np.empty((N, N))
half = N // 2
quadrants = {
    "Air": (slice(None, half), slice(None, half)),
    "CSF": (slice(None, half), slice(half, None)),
    "Fat": (slice(half, None), slice(None, half)),
    "Muscle": (slice(half, None), slice(half, None)),
}
for name, region in quadrants.items():
    density, tissue_t1, tissue_t2star = tissues[name]
    rho[region] = density
    t1[region] = tissue_t1
    t2s[region] = tissue_t2star

# Set up the images and graphs once; the slider callback updates their data.
fig, ax = plt.subplots(1, 3, figsize=(14, 10))
plt.subplots_adjust(left=0.07, right=0.95, top=0.96, bottom=0.68, wspace=0.5)
phantom = ax[0].imshow(np.zeros_like(rho), cmap='gray', vmin=0, vmax=1, origin='upper')
ax[0].set_title('Predicted tissue signal')
for xx, yy, label in [(7,7,'Air'), (23,7,'CSF'), (7,23,'Fat'), (23,23,'Muscle')]:
    ax[0].text(xx, yy, label, color='cyan', ha='center')
kim = ax[1].imshow(np.zeros((N,N)), cmap='magma', origin='lower', aspect='auto')
ax[1].set_title('Measured k-space (log magnitude)')
ax[1].set_xlabel('Readout sample index'); ax[1].set_ylabel('Phase-encode index')
im = ax[2].imshow(np.zeros((N,N)), cmap='gray', vmin=0, vmax=1, origin='upper')
ax[2].set_title('Reconstructed magnitude')
for a in (ax[0], ax[2]):
    a.set_xlabel('Horizontal position (pixel)')
    a.set_ylabel('Vertical position (pixel)')
fig.colorbar(phantom, ax=ax[0], fraction=.046, pad=.04, label='Signal magnitude (a.u.)')
kbar = fig.colorbar(kim, ax=ax[1], fraction=.046, pad=.04, label='log(1 + |signal| / 1 a.u.)')
fig.colorbar(im, ax=ax[2], fraction=.046, pad=.04, label='Signal magnitude (a.u.)')
echo_ax = fig.add_axes([.09, .40, .38, .19])
recovery_ax = fig.add_axes([.58, .40, .38, .19])
status = fig.text(.08, .33, '', fontsize=9)
specs = [('TR (ms)', 20, 2000, 2000), ('TE (ms)', 3, 100, 4),
         ('B1 (µT)', 0, 20, 5.87), ('Gx (mT/m)', 0.5, 10, 0.783),
         ('Gy max (mT/m)', 0.5, 10, 1.566)]
sliders = []
for i, (label, lo, hi, init) in enumerate(specs):
    a = fig.add_axes([.23, .28-i*.05, .62, .022])
    sliders.append(Slider(a, label, lo, hi, valinit=init))

def update(_=None):
    """Recalculate the forward model and redraw when a slider moves."""
    # TR is the time between RF pulses; TE is the time from a pulse to echo.
    # B1 controls pulse strength; Gx and Gy control spatial encoding.
    tr, te, b1, gx, gy = [s.val for s in sliders]
    tr *= 1e-3
    te *= 1e-3
    # The RF pulse tips magnetization away from the main field. For a
    # rectangular pulse, flip angle = 2*pi*gamma*B1*pulse duration.
    alpha = 2*np.pi*GAMMA_HZ*b1*1e-6*PULSE_S
    # This equation assumes repeated pulses have reached a steady state and
    # leftover transverse magnetization is removed between repetitions.
    # The exponential in TE models T2* signal loss before the echo.
    e1 = np.exp(-tr/t1)
    steady_state = (1 - e1) * np.sin(alpha) / (1 - e1 * np.cos(alpha))
    transverse0 = rho * steady_state
    signal = transverse0 * np.exp(-te / t2s)
    phantom.set_data(np.abs(signal))
    # Rectangular readout centered at TE and phase-encode blips before readout.
    # Each row has a different Gy amplitude, swept from -Gy max to +Gy max.
    kx = GAMMA_HZ*gx*1e-3*np.linspace(-READOUT_S/2, READOUT_S/2, N, endpoint=False)
    ky = GAMMA_HZ*gy*1e-3*PHASE_S*np.linspace(-1, 1, N, endpoint=False)
    # Each complex exponential is a Fourier basis function. The gradient
    # settings determine which spatial frequencies kx and ky are measured.
    ex = np.exp(-2j*np.pi*np.outer(kx, coords))
    ey = np.exp(-2j*np.pi*np.outer(ky, coords))
    # Matrix multiplication sums contributions from every pixel. The factor
    # 1/N gives the same normalization as a 2D orthonormal FFT.
    kspace = ey @ signal @ ex.T / N
    # Standard FFT reconstruction assumes the nominal Cartesian grid. When
    # gradients change, this deliberately shows the resulting FOV mismatch.
    recon = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace), norm='ortho'))
    # Log scaling makes both large and small k-space magnitudes visible.
    kim.set_data(np.log1p(np.abs(kspace)))
    kim.set_clim(0, max(np.log1p(np.abs(kspace)).max(), 1e-9))
    im.set_data(np.abs(recon))
    # Graph the signal while the central k-space line is read. The gradient
    # sweeps kx through zero at TE; all pixel signals add at each time.
    # This graph includes T2* decay at each time. For simplicity, the image
    # calculation above instead uses a single decay value at TE.
    echo_time = te + np.linspace(-READOUT_S/2, READOUT_S/2, 201)
    k_read = GAMMA_HZ * gx * 1e-3 * (echo_time - te)
    echo = np.empty(echo_time.size, complex)
    for j, (time, kval) in enumerate(zip(echo_time, k_read)):
        echo[j] = np.mean(transverse0 * np.exp(-max(time, 0)/t2s)
                          * np.exp(-2j*np.pi*kval*x))
    echo_ax.clear()
    echo_ax.plot(echo_time*1000, np.abs(echo), color='tab:blue', label='Readout magnitude')
    echo_ax.axvline(te*1000, color='tab:red', linestyle='--', label='TE')
    echo_ax.set_xlim(max(0, (te-READOUT_S/2)*1000), (te+READOUT_S/2)*1000)
    echo_ax.set_ylim(bottom=0)
    echo_ax.set_title('Gradient echo: central phase encode')
    echo_ax.set_xlabel('Time after RF pulse (ms)')
    echo_ax.set_ylabel('Signal magnitude (a.u.)')
    echo_ax.legend(loc='upper right', fontsize=8)

    # After a pulse, longitudinal magnetization recovers exponentially toward
    # equilibrium along the main field B0. Divide by its equilibrium value so
    # that 1 means fully recovered. Air is omitted because rho is zero.
    recovery_ax.clear()
    time = np.linspace(0, max(tr, te)*1.05, 400)
    for name, color in [('CSF','tab:blue'), ('Fat','tab:orange'), ('Muscle','tab:green')]:
        _, tissue_t1, _ = tissues[name]
        e = np.exp(-tr/tissue_t1)
        mz_before = (1-e)/(1-e*np.cos(alpha))
        mz = 1 + (mz_before*np.cos(alpha)-1)*np.exp(-time/tissue_t1)
        recovery_ax.plot(time*1000, mz, label=name, color=color)
    recovery_ax.axvline(te*1000, color='tab:red', linestyle='--', label='TE')
    recovery_ax.axvline(tr*1000, color='tab:purple', linestyle=':', label='TR')
    recovery_ax.axhline(1, color='gray', linewidth=.8)
    recovery_ax.set_xlim(0, max(tr, te)*1050)
    recovery_ax.set_title('Longitudinal recovery toward equilibrium')
    recovery_ax.set_xlabel('Time after RF pulse (ms)')
    recovery_ax.set_ylabel('Longitudinal magnetization / equilibrium')
    recovery_ax.legend(loc='lower right', fontsize=8, ncol=2)
    # Warn if the assumed RF pulse, phase encoding and readout cannot fit.
    # A warning does not stop the simplified mathematical calculation.
    valid = te >= PULSE_S/2 + PHASE_S + READOUT_S/2 and tr >= te + READOUT_S/2
    status.set_text(f'Flip angle: {np.degrees(alpha):.1f}°   |   '
                    f'kx range: ±{max(abs(kx)):.1f} cycles/m   |   '
                    f'ky range: ±{max(abs(ky)):.1f} cycles/m   |   '
                    + ('Timing feasible' if valid else 'Timing infeasible: increase TE or TR'))
    status.set_color('black' if valid else 'red')
    fig.canvas.draw_idle()

# Connect each slider to the same update function, then draw the first frame.
for slider in sliders:
    slider.on_changed(update)
update()
plt.show()
