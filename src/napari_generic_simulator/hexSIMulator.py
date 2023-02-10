"""
Two child classes to simulate raw data of HexSim.
"""
__author__ = "Meizhu Liang @Imperial College London"

from .baseSIMulator import Base_simulator
import numpy as np


class HexSim_simulator(Base_simulator):
    '''
    Implements hexagonal SIM illumination with three beams, seven phase steps, with beams being at 2π/3 angles.

    '''

    def __init__(self):
        self._phaseStep = 7
        self._angleStep = 1
        super().__init__()
        # systematic errors, could be arbitrary
        if self.add_error:
            self.phase_error = [0.57416431, 0.73098212, 0.67951188, 0.72925068, 0.88096275, 0.26103804, 0.7353261]
        else:
            self.phase_error = np.zeros(7)

    """All polarisations are normalised to average intensity of 1, and with theta being  π/2 for the light sheet"""

    def _illCi(self, pstep, astep):
        # Circular polarisation
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep
        _illCi = 2 / 3 + 1 / 9 * (np.cos(self.ph * np.sqrt(3) / 2 * (-np.sqrt(3) * self.x - self.y) / 2 + self._p2)
                              + np.cos(self.ph * np.sqrt(3) / 2 * (-np.sqrt(3) * self.x + self.y) / 2 + self._p1)
                              + np.cos(self.ph * np.sqrt(3) / 2 * self.y + self._p1 - self._p2))
        return _illCi

    def _illAx(self, pstep, astep):
        # Axial polarisation
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep
        _illAx = 1 / 3 + 2 / 9 * (np.cos(self.ph * np.sqrt(3) / 2 * self.y + self._p1 - self._p2)
                              + np.cos(self.ph * np.sqrt(3) / 2 * (self.y - np.sqrt(3) * self.x) / 2 + self._p1)
                              + np.cos(self.ph * np.sqrt(3) / 2 * (-self.y - np.sqrt(3) * self.x) / 2 + self._p2))
        return _illAx

    def _illIp(self, pstep, astep):
        # In-plane polarisation
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep
        _illIp = 1 / 2 - 1 / 6 * (np.cos(self.ph * np.sqrt(3) / 2 * self.y + self._p1 - self._p2)
                              + np.cos(self.ph * np.sqrt(3) / 2 * (self.y - np.sqrt(3) * self.x) / 2 + self._p1)
                              + np.cos(self.ph * np.sqrt(3) / 2 * (self.y + np.sqrt(3) * self.x) / 2 + self._p2))
        return _illIp


class RightHexSim_simulator(Base_simulator):
    '''
    Implements hexagonal SIM illumination with three beams, seven phase steps, with beams being at right angles.
    '''

    def __init__(self):
        self._phaseStep = 7
        self._angleStep = 1
        super().__init__()

    def _illCi(self, pstep, astep):
        # Circular polarisation
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep
        _illCi = 3 / 5 + 1 / 5 * (np.cos(self.ph * (-self.x + self.y) / 2 + self._p1) +
                              np.cos(self.ph * (self.x + self.y) / 2 + self._p1 - self._p2))
        return _illCi

    def _illAx(self, pstep, astep):
        # Axial polarisation with theta being π/2
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep
        _illAx = 1 / 3 + 2 / 9 * (np.cos(self.ph * (-self.x) + self._p2) +
                              np.cos(self.ph * (-self.x + self.y) / 2 + self._p1) +
                              np.cos(self.ph * (self.x + self.y) / 2 + self._p1 - self._p2))
        return _illAx

    def _illIp(self, pstep, astep):
        # In-plane polarisation
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep
        _illIp = 3 / 5 - 2 / 5 * np.cos(self.ph * (-self.x) + self._p2)
        return _illIp


if __name__ == '__main__':
    s = HexSim_simulator()
    s.acc = 3
    for msg in s.raw_image_stack():
        print(msg)
