from openbeam.backend import set_backend, to_numpy
from openbeam.core.beam import Beam
from openbeam.core.propagator import Propagator

beam = Beam(wavelength=1550e-9, size=512, physical_size=5e-3)
beam.initialize_gaussian(waist=0.5e-3)
prop = Propagator(beam)

set_backend("cpu")
prop._precompute_k_vectors()
cpu = prop.benchmark(n_runs=20)

set_backend("cuda")
prop._precompute_k_vectors()
gpu = prop.benchmark(n_runs=20)

print(f"CPU: {cpu['avg_ms']} ms")
print(f"GPU: {gpu['avg_ms']} ms")
print(f"Speedup: {cpu['avg_ms'] / gpu['avg_ms']:.1f}x faster on GPU")