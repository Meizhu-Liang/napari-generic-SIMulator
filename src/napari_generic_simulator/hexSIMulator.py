"""
Two child classes to simulate raw data of HexSim.
"""
__author__ = "Meizhu Liang @Imperial College London"

from napari_generic_simulator.baseSIMulator import Base_simulator
import numpy as np


class HexSim_simulator(Base_simulator):
    '''
    Implements hexagonal SIM illumination with three beams, seven phase steps, with beams being at 2π/3 angles.

    '''

    def __init__(self):
        self._phaseStep = 7
        self._angleStep = 1
        super().__init__()

    """All polarisations are normalised to average intensity of 1, and with theta being  π/2 for the light sheet"""
    def _illCi(self):
        # Circular polarisation
        _illCi = 1 + 1 / 6 * (np.cos(self.ph * np.sqrt(3) / 2 * (-np.sqrt(3) * self.y - self.x) / 2 + self.p2) \
                      + np.cos(self.ph * np.sqrt(3) / 2 * (-np.sqrt(3) * self.y + self.x) / 2 + self.p1) \
                      + np.cos(self.ph * np.sqrt(3) / 2 * self.x + self.p1 - self.p2))
        return [_illCi]

    def _illAx(self):
        # Axial polarisation
        _illAx = 1 + 2 / 3 * (np.cos(self.ph * np.sqrt(3) / 2 *  self.x + self.p1 - self.p2) \
                      + np.cos(self.ph * np.sqrt(3) / 2 * (self.x - np.sqrt(3) * self.y) / 2 + self.p1) \
                      + np.cos(self.ph * np.sqrt(3) / 2 * (-self.x - np.sqrt(3) * self.y) / 2 + self.p2))
        return [_illAx]

    def _illIp(self):
        # In-plane polarisation
        _illIp = 1 - 1 / 3 * (np.cos(self.ph * np.sqrt(3) / 2 * self.x + self.p1 - self.p2) \
                      + np.cos(self.ph * np.sqrt(3) / 2 * (self.x - np.sqrt(3) * self.y) / 2 + self.p1) \
                      + np.cos(self.ph * np.sqrt(3) / 2 * (self.x + np.sqrt(3) * self.y) / 2 + self.p2))
        return [_illIp]

class RightHexSim_simulator(Base_simulator):
    '''
    Implements hexagonal SIM illumination with three beams, seven phase steps, with beams being at right angles.
    '''

    def __init__(self):
        self._phaseStep = 7
        self._angleStep = 1
        super().__init__()

    def _illCi(self):
        # Circular polarisation
        _illCi = 1 + 1 / 3 * (np.cos(self.ph * (-self.y + self.x) + self.p1) +
                                   np.cos(self.ph * (self.y + self.x) + self.p1 - self.p2))
        return [_illCi]

    def _illAx(self):
        # Axial polarisation with theta being π/2
        _illAx = 1 + 2 / 3 * (np.cos(self.ph * self.x + self.p1 - self.p2) +
                                   np.cos(self.ph * (self.y + self.x) / 2 + self.p1) +
                                   np.cos(self.ph * (-self.y + self.x) / 2 + self.p2))
        return [_illAx]

    def _illIp(self):
        # In-plane polarisation
        _illIp = 1 - 2 / 3 * np.cos(self.ph * (-self.y) + self.p2)
        return [_illIp]


if __name__ == '__main__':
    s = HexSim_simulator()
    s.acc = 3
    for msg in s.raw_image_stack():
        print(msg)
