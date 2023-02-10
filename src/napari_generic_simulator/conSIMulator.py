"""
A child class to simulate raw data of conventional Sim.
"""
__author__ = "Meizhu Liang @Imperial College London"

from .baseSIMulator import Base_simulator
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
        # systematic errors, could be arbitrary
        if self.add_error:
            self.phase_error = [0.51674218, 0.76096741, 0.05281238]
            self.angle_error = [0.78095176, 0.89493163, 0.11358182]
        else:
            self.phase_error = np.zeros(3)
            self.angle_error = np.zeros(3)

    """All polarisations are normalised to average intensity of 1, and with theta being  π/2 for the light sheet"""

    def _illCi(self):
        # illumination with circular polarisation in 3 angles
        _illCi = 1
        return _illCi

    def _illAx(self, pstep, astep):
        # illumination with axial polarisation in 3 angles
        # phase
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase_error[pstep]
        # angle
        # xr, yr - Cartesian coordinate system with rotation of axes
        xr = self.xc * np.cos(astep * 2 * np.pi / self._angleStep) + self.yc * np.sin(astep * 2 * np.pi / self._angleStep)
        yr = -self.xc * np.sin(astep * 2 * np.pi / self._angleStep) + self.yc * np.cos(astep * 2 * np.pi / self._angleStep)
        _illAx = 1 / 2 + 1 / 2 * np.cos(self.ph * (xr * self.x + yr * self.y) + self._p1 + self.angle_error[astep])
        return _illAx

    def _illIp(self, pstep, astep):
        # illumination with in-plane polarisation in 3 angles
        # phase
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase_error[pstep]
        # angle
        # xr, yr - Cartesian coordinate system with rotation of axes
        xr = self.xc * np.cos(astep * 2 * np.pi / self._angleStep) + self.yc * np.sin(
            astep * 2 * np.pi / self._angleStep)
        yr = -self.xc * np.sin(astep * 2 * np.pi / self._angleStep) + self.yc * np.cos(
            astep * 2 * np.pi / self._angleStep)
        _illIp = 1 / 2 - 1 / 2 * np.cos(self.ph * (xr * self.x + yr * self.y) + self._p1 + self.angle_error[astep])
        return _illIp
