import matplotlib.pyplot as plt
import numpy as np
from openbeam.core.beam import Beam
from openbeam.components.mzi import MZI

# 1. setup: standard 1550nm beam
laser = Beam(wavelength=1550e-9, size=512, physical_size=5e-3)
laser.initialize_gaussian(waist=1.0e-3)

# 2. setup mzi
modulator = MZI()

# 3. sweep phase shift from 0 to 2*pi
phases = np.linspace(0, 2*np.pi, 20)
peak_intensities = []

print("Running phase sweep...")

for phi in phases:
    # reset beam for each step (simulating a fresh pulse)
    laser.initialize_gaussian(waist=1.0e-3)
    
    # apply mzi: arm 1 is fixed (0), arm 2 sweeps (phi)
    modulator.apply(laser, phase_arm1=0, phase_arm2=phi)
    
    # record peak intensity
    peak_intensities.append(np.max(laser.intensity))

# 4. plot the results (the characteristic cosine squared curve)
plt.figure(figsize=(8, 5))
plt.plot(phases, peak_intensities, 'o-', color='red', label='Simulation')

# theoretical curve for comparison
theory = np.cos(phases/2)**2
plt.plot(phases, theory, '--', color='black', alpha=0.5, label='Theory (Cos^2)')

plt.title("MZI Modulation Transfer Function")
plt.xlabel("Phase Difference (Radians)")
plt.ylabel("Output Intensity")
plt.grid(True)
plt.legend()
plt.show()