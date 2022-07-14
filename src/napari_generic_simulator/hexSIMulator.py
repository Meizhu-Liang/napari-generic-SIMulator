"""
Two child classes to simulate raw data of HexSim. @author: Meizhu Liang @Imperial College
"""

from napari_generic_simulator.baseSIMulator import Base_simulator
import numpy as np


class HexSim_simulator(Base_simulator):
    '''
    Implements hexagonal SIM illumination with three beams, seven phase steps, with beams being at 2π/3 angles.

    eta is the factor by which the illumination grid frequency exceeds the incoherent cutoff, eta = 1 for normal
    SIM, eta=sqrt(3) / 2 to maximise resolution without zeros in TF.
    For a normal SIM, maximum resolution extension = 1 + eta
    carrier is 2 * kmax * eta
    '''

    def __init__(self):
        self._phaseStep = 7
        self._angleStep = 1
        super().__init__()
        self.eta = self.n * np.sqrt(3.0) / 2 / self.NA

    def _ill(self):
        """All polarisations are normalised to average intensity of 1, and with theta being  π/2 for the light sheet"""
        # Circular polarisation
        self._illCi = 1 + 1 / 6 * (np.cos(self.ph * (-np.sqrt(3) * self.x - self.y) / 2 + self.p2) \
                      + np.cos(self.ph * (-np.sqrt(3) * self.x + self.y) / 2 + self.p1) \
                      + np.cos(self.ph * self.y + self.p1 - self.p2))
        # Axial polarisation
        self._illAx = 1 + 2 / 3 * (np.cos(self.ph * self.y + self.p1 - self.p2) \
                      + np.cos(self.ph * (self.y - np.sqrt(3) * self.x) / 2 + self.p1) \
                      + np.cos(self.ph * (-self.y - np.sqrt(3) * self.x) / 2 + self.p2))
        # In-plane polarisation
        self._illIp = 1 - 1 / 3 * (np.cos(self.ph * self.y + self.p1 - self.p2) \
                      - np.cos(self.ph * (self.y - np.sqrt(3) * self.x) / 2 + self.p1) \
                      - np.cos(self.ph * (-self.y - np.sqrt(3) * self.x) / 2 + self.p2))

class RightHexSim_simulator(Base_simulator):
    '''
    Implements hexagonal SIM illumination with three beams, seven phase steps, with beams being at right angles.
    '''

    def __init__(self):
        self._phaseStep = 7
        self._angleStep = 1
        super().__init__()
        self.eta = self.n / self.NA

    def _ill(self):
        # Axial polarisation with theta being π/2
        self._illAx = 1 + 2 / 3 * (np.cos(self.ph * (self.x) + self.p2) + np.cos(self.ph * (self.x + self.y) / 2 +
                                    self.p1 - self.p2)+ np.cos(self.ph * (-self.x + self.y) / 2 + self.p1))
        # In-plane polarisation
        self._illIp = 1 - 2 / 3 * np.cos(self.ph * (-self.x) + self.p2)
        # Circular polarisation
        self._illCi = 1 + 1 / 3 * (np.cos(self.ph * (-self.x + self.y) + self.p1) +
                                    np.cos(self.ph * (self.x + self.y) + self.p1 - self.p2))


if __name__ == '__main__':
    s = HexSim_simulator()
    s.acc = 3
    t = s.raw_image_stack()
    try:
        while True:
            print(next(t))
    except Exception as e:
        print(e)