"""
Classes to calculate SIM illumination.
"""
__author__ = "Meizhu Liang @Imperial College London"

from .baseSIMulator import Base_simulator
import numpy as np
from numpy import cos, sin, pi


class Illumination(Base_simulator):
    """
    A class to calculate illumination patterns of multiple beams.
    """

    def __init__(self):
        super().__init__()
        self._nsteps = int(self._phaseStep * self._angleStep)
        self._nbands = int((self._nsteps - self._angleStep) / 2)
        self._beam_a = 2 * np.pi / self._n_beams  # angle between each two beams


        # S_beams: Jones vector; E_beams: exponential term; a beam could be expressed as S_beams @ E_beams
        self.S_beams = np.complex64(np.zeros((self._angleStep, self._n_beams, 3)))


    def rotation(self, phi, theta):
        """Calculates the rotation matrix"""
        R = np.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]) \
                 @ np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]]) \
                 @ np.array([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
        return R

    def _get_alpha_constants(self):
        self.theta = np.arcsin(self.ill_NA / self.n)
        self.alpha_matrix = np.complex64(np.zeros((self.npoints, self._angleStep, self._phaseStep)))
        con = np.complex64(np.zeros((self._angleStep, self._n_beams)))  # constant alpha values

        # get alpha values
        for a in range(self._angleStep):
            for i in range(self._n_beams):
                phi = i * self._beam_a + a * 2 * np.pi / self._angleStep
                # rotation matrix for the field travelling in z, not for illumination patterns.
                # phi is the azimuthal angle (0 - 2pi). theta is the polar angle (0 - pi).

                self.S_beams[a, i, :] = self.rotation(phi, self.theta) @ np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)], [0, 0]]) @ self.f_p
                con[a, i] = self.S_beams[a, i] @ np.conjugate(self.S_beams[a, i])
            self.alpha_matrix[:, a, 0] = np.sum(con[a])  # constant alpha values

    def _get_alpha(self, x, y, astep):
        self.E_beams = np.complex64(np.zeros((self.npoints, self._angleStep, self._n_beams, 3)))
        self.alpha_band = np.complex64(np.zeros((self.npoints, self._angleStep, self._nbands)))
        xyz = np.transpose(np.array([x, y, np.zeros(self.npoints)]))

        for i in range(self._n_beams):
            phi = i * self._beam_a + astep * 2 * np.pi / self._angleStep
            print('here!!!!!!!!!!!!!!!!!')
            print((xyz @ self.rotation(phi, self.theta)).shape, np.transpose(np.array([0, 0, self.k0])).shape, (self.E_beams[:, astep, i, :]).shape)
            self.E_beams[:, astep, i, :] = np.exp(-1j * (xyz @ self.rotation(phi, self.theta) @ np.transpose(np.array([0, 0, self.k0]))))
            print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        b = 0
        for i in range(self._n_beams):
            for j in range(int(self._n_beams - i - 1)):

                self.alpha_band[:, astep, b] = self.S_beams[astep, i] @ np.conj(
                    self.S_beams[astep, i + j + 1] * self.E_beams[:, astep, i] * np.conjugate(self.E_beams[:, astep, i + j + 1]))
                print('aaaaaaaaaaaaaaaaaaaaaaaaaa')
                b += 1
        for i in range((self._phaseStep - 1) // 2):
            self.alpha_matrix[:, astep, i + 1] = self.alpha_band[:, astep, i]
            self.alpha_matrix[:, astep, i + 1 + b] = np.conjugate(self.alpha_band[:, astep, i])
        return self.alpha_matrix

    def _get_phases(self, pstep):
        Phi0 = 2 * np.pi / self._phaseStep
        self.phase_matrix = np.complex64(np.ones(int(self._nbands / self._angleStep * 2 + 1)))
        for i in range(int(self._nbands / self._angleStep)):
            self.phase_matrix[i+1] = np.exp(1j * (pstep * (i + 1) * Phi0))
            self.phase_matrix[int(i + 1 + self._nbands / self._angleStep)] = np.exp(-1j * (pstep * (i + 1) * Phi0))
        return self.phase_matrix

    def _ill_test(self, x, y, pstep, astep):
        return self._get_phases(pstep) @ self._get_alpha(x, y, astep)[:, astep, :]


class ConIll(Illumination):
    def __init__(self):
        self._phaseStep = 3
        self._angleStep = 3
        self._n_beams = 2
        super().__init__()

class HexIll(Illumination):
    def __init__(self):
        self._phaseStep = 7
        self._angleStep = 1
        self._n_beams = 3
        super().__init__()
