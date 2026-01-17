import numpy as np
from openbeam.core.beam import Beam

class Lens:
    # represents a thin lens that applies a quadratic phase shift
    
    def __init__(self, focal_length: float):
        # focal_length (float): distance to the focal point in meters
        self.focal_length = focal_length

    def apply(self, beam: Beam):
        # applies the lens phase transmission function to the beam
        # t(x,y) = exp(-i * k / (2f) * (x^2 + y^2))
        
        k = 2 * np.pi / beam.wavelength
        r_squared = beam.X**2 + beam.Y**2
        
        # calculate phase shift
        phase_shift = np.exp(-1j * k / (2 * self.focal_length) * r_squared)
        
        # update the beam field
        beam.field *= phase_shift