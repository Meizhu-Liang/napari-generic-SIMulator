"""
Two child classes to simulate raw data of HexSim. @author: Meizhu Liang @Imperial College
"""

from napari_sim_simulator.baseSIMulator import Base_simulator
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
        self._nsteps = 7
        super().__init__()
        self._eta = self.n * np.sqrt(3.0) / 2 / self.NA

    def _ill(self):
        # Axial polarisation normalised to peak intensity of 1
        self._illAx = 1 / 9 * (6 + np.cos(self.ph * self.y + self.p1 - self.p2) + np.cos(self.ph * (self.y - np.sqrt(3) * self.x) / 2 + self.p1) +
                                   np.cos(self.ph * (-self.y - np.sqrt(3) * self.x) / 2 + self.p2))
        # In-plane polorisation
        self._illIp = 1 / 6 * (3 - np.cos(self.ph * self.y + self.p1 - self.p2) - np.cos(self.ph * (self.y - np.sqrt(3) * self.x) / 2 + self.p1)
                    - np.cos(self.ph * (-self.y - np.sqrt(3) * self.x) / 2 + self.p2))

class RightHexSim_simulator(Base_simulator):
    '''
    Implements hexagonal SIM illumination with three beams, seven phase steps, with beams being at right angles.
    '''

    def __init__(self):
        self._nsteps = 7
        super().__init__()
        self._eta = self.n / self.NA

    def _ill(self):
        # Axial polarisation normalised to peak intensity of 1, with theta being π/2
        self._illAx = 1 / 9 * (3 + 2 * np.cos(self.ph * (-2 * self.x) + self.p2) + 2 * np.cos(self.ph * (-self.y - self.x) + self.p1 +self.p2) +
                                   2 * np.cos(self.ph * (self.x - self.y) + self.p1))
        # In-plane polarisation normalised to peak intensity of 1
        self._illIp = 1 / 5 * (3 - 2 * np.cos(self.ph * (-2 * self.x) + self.p2))

if __name__=='__main__':
    tt = HexSim_simulator()
    tt.raw_image_stack()


