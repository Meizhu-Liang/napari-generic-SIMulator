"""
A child class to simulate raw data of conventional Sim. @author: Meizhu Liang @Imperial College
"""

from napari_generic_simulator.baseSIMulator import Base_simulator
import numpy as np

class ConSim_simulator(Base_simulator):
    '''
    Implements sinusoidal SIM illumination with two beams, three angles and three phase steps.

    eta is the factor by which the illumination grid frequency exceeds the incoherent cutoff, eta = 1 for normal
    SIM, eta=sqrt(3) / 2 to maximise resolution without zeros in TF.
    For a normal SIM, maximum resolution extension = 1 + eta
    carrier is 2 * kmax * eta
    '''

    def __init__(self):
        self._phaseStep = 3
        self._angleStep = 3
        super().__init__()
        self.eta = self.n / self.NA

    def _ill(self):
        """All polarisations are normalised to average intensity of 1, and with theta being  π/2 for the light sheet"""
        # Circular polarisation
        self._illCi = 1
        # Axial polarisation
        self._illAx = 1 + np.cos(self.ph * (-2 * self.x) + self.p1)
        # In-plane polarisation
        self._illIp = 1 - np.cos(self.ph * (-2 * self.x) + self.p1)