import numpy as np
from openbeam.core.beam import Beam

class MZI:
    # represents a single mach-ehnder interferometer acting as a modulator
    # used for controlling beam amplitude and phase via interference
    
    def __init__(self):
        # assumes a balanced 50:50 splitter setup initially
        pass

    def apply(self, beam: Beam, phase_arm1: float, phase_arm2: float):
        # applies the mzi transfer function to the input beam
        # args:
        # phase_arm1 (float): phase shift in the upper arm (radians)
        # phase_arm2 (float): phase shift in the lower arm (radians)
        
        # calculate the interference factor
        # e_out = e_in * 0.5 * (exp(i*phi1) + exp(i*phi2))
        interference_term = 0.5 * (np.exp(1j * phase_arm1) + np.exp(1j * phase_arm2))
        
        # update the beam field
        beam.field *= interference_term
        
    def get_extinction_ratio(self, phase_diff: float) -> float:
        # helper to calculate theoretical transmission for a given phase difference
        return np.cos(phase_diff / 2)**2