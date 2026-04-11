import numpy as np

from openbeam.core.beam import Beam
from openbeam.backend import get_array_module, to_numpy, current_backend


class Propagator:
    def __init__(self, beam: Beam):
        self.beam = beam
        self._precompute_k_vectors()

    def _precompute_k_vectors(self):
        xp = get_array_module()
        kx = xp.fft.fftfreq(self.beam.size, d=self.beam.dx) * 2 * np.pi
        self.KX, self.KY = xp.meshgrid(kx, kx)
        k0 = 2 * np.pi / self.beam.wavelength
        self.KZ = xp.sqrt(k0**2 - self.KX**2 - self.KY**2 + 0j)

    def propagate(self, distance: float):
        xp = get_array_module()
        field = xp.asarray(self.beam.field)
        field_fft = xp.fft.fft2(field)
        transfer_function = xp.exp(1j * self.KZ * distance)
        propagated_fft = field_fft * transfer_function
        self.beam.field = to_numpy(xp.fft.ifft2(propagated_fft))

    def propagate_z_stack(self, distances: list) -> np.ndarray:
        xp = get_array_module()
        field = xp.asarray(self.beam.field)
        field_fft = xp.fft.fft2(field)
        results = []
        for z in distances:
            H = xp.exp(1j * self.KZ * z)
            results.append(to_numpy(xp.fft.ifft2(field_fft * H)))
        return np.stack(results, axis=0)

    def benchmark(self, n_runs: int = 20) -> dict:
        import time
        xp = get_array_module()
        self.propagate(0.1)
        if current_backend() == "cuda":
            xp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.propagate(0.1)
        if current_backend() == "cuda":
            xp.cuda.Stream.null.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000
        return {
            "backend": current_backend(),
            "n_runs": n_runs,
            "total_ms": round(total_ms, 2),
            "avg_ms": round(total_ms / n_runs, 2),
        }