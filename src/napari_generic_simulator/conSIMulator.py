"""
A child class to simulate raw data of conventional Sim.
"""
__author__ = "Meizhu Liang @Imperial College London"

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
        self._eta = 0.5 * self.n / self.NA

    """All polarisations are normalised to average intensity of 1, and with theta being  π/2 for the light sheet"""

    def _illCi(self):
        # illumination with circular polarisation in 3 angles
        _illCi_0 = 1
        _illCi_1 = 1
        _illCi_2 = 1
        return _illCi_0, _illCi_1, _illCi_2

    def _illAx(self):
        # illumination with axial polarisation in 3 angles
        _illAx_0 = 1 + 1 / 3 * np.cos(self.ph * (-2 * self.x) + self.p1)
        _illAx_1 = 1 + 1 / 3 * np.cos(self.ph * (self.x - np.sqrt(3) * self.y) + self.p1)
        _illAx_2 = 1 + 1 / 3 * np.cos(self.ph * (self.x + np.sqrt(3) * self.y) + self.p1)
        return _illAx_0, _illAx_1, _illAx_2

    def _illIp(self):
        # illumination with in-plane polarisation in 3 angles
        _illIp_0 = 1 - 1 / 3 * np.cos(self.ph * (-2 * self.x) + self.p1)
        _illIp_1 = 1 - 1 / 3 * np.cos(self.ph * (self.x - np.sqrt(3) * self.y) + self.p1)
        _illIp_2 = 1 - 1 / 3 * np.cos(self.ph * (self.x + np.sqrt(3) * self.y) + self.p1)
        return _illIp_0, _illIp_1, _illIp_2
