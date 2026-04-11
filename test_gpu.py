from openbeam.backend import set_backend, to_numpy
from openbeam.core.beam import Beam
from openbeam.core.propagator import Propagator
import matplotlib.pyplot as plt

# --- choose "cuda" if you have a GPU, "cpu" otherwise ---
set_backend("cuda")

# set up a beam
beam = Beam(wavelength=1550e-9, size=512, physical_size=5e-3)
beam.initialize_gaussian(waist=0.5e-3)

# propagate
prop = Propagator(beam)
prop.propagate(distance=0.1)

# benchmark
result = prop.benchmark(n_runs=20)
print(result)

# plot
plt.imshow(to_numpy(beam.intensity), cmap="inferno")
plt.title(f"Propagated beam [{result['backend'].upper()}] — {result['avg_ms']} ms/call")
plt.colorbar()
plt.show()