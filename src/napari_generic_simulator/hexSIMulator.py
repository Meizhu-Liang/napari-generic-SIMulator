"""
Two child classes to simulate raw data of HexSim.
"""
__author__ = "Meizhu Liang @Imperial College London"

from .baseSIMulator import Base_simulator
import numpy as np
from numpy import cos, sin


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
            self.phase1_error = 2 * np.random.random(7) - 1
            self.phase1_error[0] = 0.0
            self.phase2_error = 2 * np.random.random(7) - 1
            self.phase2_error[0] = 0.0
            print(self.phase1_error)
            print(self.phase2_error)
        else:
            self.phase1_error = np.zeros(7)
            self.phase2_error = np.zeros(7)

    """All polarisations are normalised to average intensity of 1, and with theta being  π/2 for the light sheet"""

    def _illCi(self, pstep, astep):
        # Circular polarisation
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase1_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep + self.phase2_error[pstep]
        _illCi = 2 / 3 + 1 / 9 * (np.cos(self.ph * np.sqrt(3) / 2 * (-np.sqrt(3) * self.x - self.y) / 2 + self._p2)
                              + np.cos(self.ph * np.sqrt(3) / 2 * (-np.sqrt(3) * self.x + self.y) / 2 + self._p1)
                              + np.cos(self.ph * np.sqrt(3) / 2 * self.y + self._p1 - self._p2))
        return _illCi

    def _illAx(self, pstep, astep):
        # Axial polarisation
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase1_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep + self.phase2_error[pstep]
        _illAx = 1 / 3 + 2 / 9 * (np.cos(self.ph * np.sqrt(3) / 2 * self.y + self._p1 - self._p2)
                              + np.cos(self.ph * np.sqrt(3) / 2 * (self.y - np.sqrt(3) * self.x) / 2 + self._p1)
                              + np.cos(self.ph * np.sqrt(3) / 2 * (-self.y - np.sqrt(3) * self.x) / 2 + self._p2))
        return _illAx

    def _illIp(self, pstep, astep):
        # In-plane polarisation
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase1_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep + self.phase2_error[pstep]
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
        if self.add_error:
            self.phase1_error = np.random.random(7) - 0.5
            self.phase1_error[0] = 0.0
            self.phase2_error = np.random.random(7) - 0.5
            self.phase2_error[0] = 0.0
            print(self.phase1_error)
            print(self.phase2_error)
        else:
            self.phase1_error = np.zeros(7)
            self.phase2_error = np.zeros(7)

    def _illCi(self, pstep, astep):
        # Circular polarisation
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase1_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep + self.phase2_error[pstep]
        _illCi = 3 / 5 + 1 / 5 * (np.cos(self.ph * (-self.x + self.y) / 2 + self._p1) +
                              np.cos(self.ph * (self.x + self.y) / 2 + self._p1 - self._p2))
        return _illCi

    def _illAx(self, pstep, astep):
        # Axial polarisation with theta being π/2
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase1_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep + self.phase2_error[pstep]
        _illAx = 1 / 3 + 2 / 9 * (np.cos(self.ph * (-self.x) + self._p2) +
                              np.cos(self.ph * (-self.x + self.y) / 2 + self._p1) +
                              np.cos(self.ph * (self.x + self.y) / 2 + self._p1 - self._p2))
        return _illAx

    def _illIp(self, pstep, astep):
        # In-plane polarisation
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase1_error[pstep]
        self._p2 = -pstep * 4 * np.pi / self._phaseStep + self.phase2_error[pstep]
        _illIp = 3 / 5 - 2 / 5 * np.cos(self.ph * (-self.x) + self._p2)
        return _illIp

class Illumination(Base_simulator):
    """
    A class to calculate illumination patterns of multiple beams.
    """

    def __init__(self):
        self._phaseStep = 7
        self._angleStep = 1
        self._n_beams = 3
        self._nsteps = int(self._phaseStep * self._angleStep)
        self._nbands = int((self._nsteps - self._angleStep) / 2)
        self._beam_a = 2 * np.pi / self._n_beams  # angle between each two beams
        super().__init__()

        # f_p: field components of different polarised beams
        # axial
        self.f_p = np.array([1, 0])
        # circular
        # self.f_p = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])
        # in-plane
        # self.f_p = np.array([0, 1])

    def _get_alpha_matrix(self):
        k0 = 2 * np.pi * self.n / (self.ill_wavelength * 0.001)
        # S_beams: Jones vector; E_beams: exponential term; a beam could be expressed as S_beams @ E_beams
        S_beams, E_beams = np.complex64(np.zeros((self._n_beams, 3))), np.complex64(np.zeros((self._n_beams, 3)))
        self.alpha_matrix = np.complex64(np.zeros((self._angleStep, self._phaseStep)))
        con = np.complex64(np.zeros(self._n_beams))  # constant alpha values

        alpha_band = np.complex64(np.zeros(self._nbands))

        # get alpha values
        for a in range(self._angleStep):
            for i in range(self._n_beams):
                phi = i * self._beam_a
                theta = np.pi / 2

                # rotation matrix for the field travelling in z, not for illumination patterns.
                # phi is the azimuthal angle (0 - 2pi). theta is the polar angle (0 - pi).
                R = np.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]) \
                    @ np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]]) \
                    @ np.array([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])

                S_beams[i, :] = R @ np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)], [0, 0]]) @ self.f_p

                E_beams[i, :] = np.exp(-1j * (np.array([self.x, self.y, 0]) @ R @ np.array([0, 0, k0])))
                con[i] = S_beams[i] @ np.conjugate(S_beams[i])
            self.alpha_matrix[a, 0] = np.sum(con)  # constant alpha values

            b = 0
            for i in range(self._n_beams):
                for j in range(int(self._n_beams-i-1)):
                    alpha_band[b] = S_beams[i] @ np.conj(S_beams[i + j + 1] * E_beams[i] * np.conjugate(E_beams[i + j + 1]))
                    b += 1
            for i in range((self._phaseStep-1)//2):
                self.alpha_matrix[a, i + 1] = alpha_band[i]
                self.alpha_matrix[a, i + 1 + b] = np.conjugate(alpha_band[i])

    def _get_phases(self):
        Phi0 = 2 * np.pi / self._phaseStep
        self.phase_matrix = np.complex64(np.zeros((self._angleStep, self._phaseStep, self._phaseStep)))
        n_eff_phases = int((self._phaseStep + 1)/2)  # number of effective phases terms
        for a in range(self._angleStep):
            self.phase_matrix[a, :, 0] = 1
            for i in range(self._phaseStep):
                for j in range(1, n_eff_phases):
                    self.phase_matrix[a, i, j] = np.exp(1j * i * j * Phi0)
                    self.phase_matrix[a, i, j + n_eff_phases-1] = np.exp(-1j * i * j * Phi0)
                    print(i,j)

    def _ill_test(self, pstep, astep):
        return self.phase_matrix[astep, pstep, :] @ self.alpha_matrix[astep, :]


if __name__ == '__main__':
    s = HexSim_simulator()
    s.acc = 3
    for msg in s.raw_image_stack():
        print(msg)
