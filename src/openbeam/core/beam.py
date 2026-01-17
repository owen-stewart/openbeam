import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Beam:
    # represents the complex optical field of a laser beam
    # attributes:
    # wavelength (float): the wavelength of the light in meters (e.g., 1550e-9)
    # size (int): the number of grid points along one side (n x n grid)
    # physical_size (float): the physical width of the simulation window in meters
    wavelength: float
    size: int = 1024  # standard power of 2 for fft efficiency
    physical_size: float = 5e-3  # 5mm window

    def __post_init__(self):
        # initializes the physical grid and the empty field array
        self.dx = self.physical_size / self.size
        self.field = np.zeros((self.size, self.size), dtype=np.complex128)
        self._setup_grid()

    def _setup_grid(self):
        # creates the x and y coordinate meshes centered at (0,0)
        limit = self.physical_size / 2
        linspace = np.linspace(-limit, limit, self.size)
        self.X, self.Y = np.meshgrid(linspace, linspace)

    def initialize_gaussian(self, waist: float, amplitude: float = 1.0):
        # sets the current field to a fundamental gaussian beam (tem00)
        # args:
        # waist (float): the beam waist radius (w0) in meters
        # amplitude (float): peak electric field amplitude
        
        # gaussian equation: e(r) = a * exp(-r^2 / w0^2)
        r_squared = self.X**2 + self.Y**2
        
        # fix: strictly cast to complex128 so we don't lose the data type
        result = amplitude * np.exp(-r_squared / (waist**2))
        self.field = result.astype(np.complex128)
        
    @property
    def intensity(self) -> np.ndarray:
        # returns the intensity distribution i = |e|^2
        return np.abs(self.field)**2

    @property
    def phase(self) -> np.ndarray:
        # returns the phase distribution of the field
        return np.angle(self.field)