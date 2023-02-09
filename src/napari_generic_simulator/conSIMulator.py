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
        # xc, yc - Cartesian coordinate system
        self.xc = -1
        self.yc = 0
        super().__init__()

    """All polarisations are normalised to average intensity of 1, and with theta being  π/2 for the light sheet"""

    def _illCi(self):
        # illumination with circular polarisation in 3 angles
        _illCi = 1
        return _illCi

    def _illAx(self, pstep, astep):
        # illumination with axial polarisation in 3 angles
        # phase
        self._p1 = pstep * 2 * np.pi / self._phaseStep
        # angle
        # xr, yr - Cartesian coordinate system with rotation of axes
        xr = self.xc * np.cos(astep * 2 * np.pi / self._angleStep) + self.yc * np.sin(astep * 2 * np.pi / self._angleStep)
        yr = -self.xc * np.sin(astep * 2 * np.pi / self._angleStep) + self.yc * np.cos(astep * 2 * np.pi / self._angleStep)
        _illAx = 1 / 2 + 1 / 2 * np.cos(self.ph * (xr * self.x + yr * self.y) + self._p1)
        # _illAx_0 = 1 + 1 / 2 * np.cos(self.ph * (-2 * self.x) + self._p1)
        # _illAx_1 = 1 + 1 / 2 * np.cos(self.ph * (self.x - np.sqrt(3) * self.y) + self._p1)
        # _illAx_2 = 1 + 1 / 2 * np.cos(self.ph * (self.x + np.sqrt(3) * self.y) + self._p1)
        return _illAx

    def _illIp(self, pstep, astep):
        # illumination with in-plane polarisation in 3 angles
        # phase
        self._p1 = pstep * 2 * np.pi / self._phaseStep
        # angle
        # xr, yr - Cartesian coordinate system with rotation of axes
        xr = self.xc * np.cos(astep * 2 * np.pi / self._angleStep) + self.yc * np.sin(
            astep * 2 * np.pi / self._angleStep)
        yr = -self.xc * np.sin(astep * 2 * np.pi / self._angleStep) + self.yc * np.cos(
            astep * 2 * np.pi / self._angleStep)
        _illIp = 1 / 2 - 1 / 2 * np.cos(self.ph * (xr * self.x + yr * self.y) + self._p1)
        # _illIp_0 = 1 - 1 / 2 * np.cos(self.ph * (-2 * self.x) + self._p1)
        # _illIp_1 = 1 - 1 / 2 * np.cos(self.ph * (self.x - np.sqrt(3) * self.y) + self._p1)
        # _illIp_2 = 1 - 1 / 2 * np.cos(self.ph * (self.x + np.sqrt(3) * self.y) + self._p1)
        return _illIp
