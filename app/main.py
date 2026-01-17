import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from openbeam.core.beam import Beam
from openbeam.core.propagator import Propagator
from openbeam.components.lens import Lens
from openbeam.components.mzi import MZI
# page configuration
st.set_page_config(page_title="OpenBeam Simulator", layout="wide")

st.title("🔬 OpenBeam: Optical Design Engine")
st.markdown("Interactive simulation of free-space optics and reconfigurable photonics.")

# --- sidebar controls ---
st.sidebar.header("1. Laser Source")
wavelength = st.sidebar.number_input("Wavelength (m)", value=1550e-9, format="%.2e")
waist_mm = st.sidebar.slider("Beam Waist (mm)", 0.1, 2.0, 1.0)
waist = waist_mm * 1e-3

st.sidebar.header("2. Experiment Setup")
mode = st.sidebar.radio("Select Mode", ["Free Space Diffraction", "Lens Focusing", "MZI Modulator"])

# --- simulation logic ---

# initialize the beam
laser = Beam(wavelength=wavelength, size=512, physical_size=5e-3)
laser.initialize_gaussian(waist=waist)
engine = Propagator(laser)

if mode == "Free Space Diffraction":
    st.subheader("Free Space Propagation")
    distance_cm = st.slider("Propagation Distance (cm)", 0.0, 50.0, 10.0)
    
    # propagate
    engine.propagate(distance=distance_cm * 1e-2)
    
    # visualization
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(laser.intensity, cmap='hot', extent=[-2.5, 2.5, -2.5, 2.5])
    plt.colorbar(im, ax=ax, label="Intensity (I)")
    ax.set_title(f"Beam Profile at z = {distance_cm} cm")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    st.pyplot(fig)

elif mode == "Lens Focusing":
    st.subheader("Lens Focusing Test")
    focal_length_cm = st.slider("Lens Focal Length (cm)", 1.0, 20.0, 10.0)
    
    # 1. propagate to lens (fixed 5cm)
    engine.propagate(0.05)
    
    # 2. apply lens
    lens = Lens(focal_length=focal_length_cm * 1e-2)
    lens.apply(laser)
    
    # 3. propagate to detector
    # let user slide the detector to find the focal point
    detector_z_cm = st.slider("Detector Position (cm from lens)", 0.0, 30.0, 10.0)
    engine.propagate(detector_z_cm * 1e-2)
    
    # visualization
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(laser.intensity, cmap='hot', extent=[-2.5, 2.5, -2.5, 2.5])
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Detector at z = {detector_z_cm} cm")
    st.pyplot(fig)

elif mode == "MZI Modulator":
    st.subheader("Mach-Zehnder Interferometer Switch")
    
    col1, col2 = st.columns(2)
    with col1:
        phase_delta = st.slider("Phase Shift (Radians)", 0.0, 2*np.pi, 0.0)
    
    # apply mzi
    modulator = MZI()
    modulator.apply(laser, phase_arm1=0, phase_arm2=phase_delta)
    
    # visualization
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(laser.intensity, cmap='hot', vmin=0, vmax=1.0, extent=[-2.5, 2.5, -2.5, 2.5])
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Output Beam (Phase Delta = {phase_delta:.2f})")
    st.pyplot(fig)
    
    # show transmission stat
    transmission = np.max(laser.intensity)
    st.metric("Transmission", f"{transmission*100:.1f}%")