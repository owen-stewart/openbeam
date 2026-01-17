import matplotlib.pyplot as plt
from openbeam.core.beam import Beam
from openbeam.core.propagator import Propagator
from openbeam.components.lens import Lens

# 1. setup: 1550nm beam with a wide waist (so it's easy to focus)
laser = Beam(wavelength=1550e-9, size=1024, physical_size=5e-3)
laser.initialize_gaussian(waist=1.0e-3)

# 2. setup engine and lens (f = 10cm)
engine = Propagator(laser)
lens = Lens(focal_length=0.1)

# 3. step 1: propagate 5cm to the lens
engine.propagate(distance=0.05)

# 4. step 2: apply the lens
print("Applying Lens...")
lens.apply(laser)

# 5. step 3: propagate 10cm (the focal length)
print("Propagating to focus...")
engine.propagate(distance=0.1)

# 6. visualize the focus
plt.figure(figsize=(6, 6))
plt.imshow(laser.intensity, cmap='hot', extent=[-2.5, 2.5, -2.5, 2.5])
plt.title("Beam at Focal Point (z = f)")
plt.colorbar(label="Intensity")
plt.show()