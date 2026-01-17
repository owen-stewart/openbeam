import numpy as np
import scipy.fftpack as fft
from openbeam.core.beam import Beam

class Propagator:
    # handles the free-space propagation using the angular spectrum method
    
    def __init__(self, beam: Beam):
        self.beam = beam
        self._precompute_k_vectors()

    def _precompute_k_vectors(self):
        # calculates the spatial frequency grid (kx, ky) required for the transfer function
        
        # 1. defined frequency coordinates
        dk = 2 * np.pi / self.beam.physical_size
        kx = np.fft.fftfreq(self.beam.size, d=self.beam.dx) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, kx)

        # 2. calculate kz (longitudinal wave vector)
        # k0 = 2*pi / lambda
        k0 = 2 * np.pi / self.beam.wavelength
        
        # kz = sqrt(k0^2 - kx^2 - ky^2)
        # we use complex types to handle evanescent waves (where the term inside sqrt is negative)
        self.KZ = np.sqrt(k0**2 - self.KX**2 - self.KY**2 + 0j)

    def propagate(self, distance: float):
        # moves the beam forward by a specific distance (z)
        
        # 1. move to frequency domain (fft)
        field_fft = fft.fft2(self.beam.field)

        # 2. apply the transfer function: H(kx, ky) = exp(i * kz * z)
        transfer_function = np.exp(1j * self.KZ * distance)
        propagated_fft = field_fft * transfer_function

        # 3. move back to spatial domain (ifft)
        self.beam.field = fft.ifft2(propagated_fft)