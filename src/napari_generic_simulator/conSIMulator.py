"""
A child class to simulate raw data of conventional Sim.
"""
__author__ = "Meizhu Liang @Imperial College London"

from .baseSIMulator import Base_simulator
import numpy as np
from numpy import sin, cos


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
            self.phase_error = np.array([[0., 0.5, -0.5],
                                         [0., 0.5, 0.5],
                                         [0., -0.5, 0.5]])
            self.angle_error = np.array([0.78095176, 0.89493163, 0.11358182])
        else:
            self.phase_error = np.zeros((self._angleStep, self._phaseStep))
            self.angle_error = np.zeros(self._angleStep)

    """All polarisations are normalised to average intensity of 1, and with theta being  π/2 for the light sheet"""

    def _illCi(self, pstep: int, astep: int):
        # illumination with circular polarisation in 3 angles
        _illCi = 1
        return _illCi

    def _illAx(self, pstep: int, astep: int):
        # illumination with axial polarisation in 3 angles
        # phase
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase_error[astep, pstep]
        angle = astep * 2 * np.pi / self._angleStep + self.angle_error[astep] + np.pi / 6
        # xr, yr - Cartesian coordinate system with rotation of axes
        xr = self.xc * np.cos(angle) + self.yc * np.sin(angle)
        yr = -self.xc * np.sin(angle) + self.yc * np.cos(angle)
        _illAx = 1 / 2 + 1 / 2 * np.cos(self.ph * (xr * self.x + yr * self.y) + self._p1)
        return _illAx

    def _illIp(self, pstep: int, astep: int):
        # illumination with in-plane polarisation in 3 angles
        # phase
        self._p1 = pstep * 2 * np.pi / self._phaseStep + self.phase_error[astep, pstep]
        angle = astep * 2 * np.pi / self._angleStep + self.angle_error[astep] + np.pi / 6
        # xr, yr - Cartesian coordinate system with rotation of axes
        xr = self.xc * np.cos(angle) + self.yc * np.sin(angle)
        yr = -self.xc * np.sin(angle) + self.yc * np.cos(angle)
        _illIp = 1 / 2 - 1 / 2 * np.cos(self.ph * (xr * self.x + yr * self.y) + self._p1)
        return _illIp


class Illumination(Base_simulator):
    """
    A class to calculate illumination patterns of multiple beams.
    """

    def __init__(self):
        self._phaseStep = 3
        self._angleStep = 3
        self._n_beams = 2
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

        alpha_band = np.complex64(np.zeros(self._angleStep, self._nbands))

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
                    alpha_band[a, b] = S_beams[i] @ np.conj(S_beams[i + j + 1] * E_beams[i] * np.conjugate(E_beams[i + j + 1]))
                    b += 1
            for i in range((self._phaseStep-1)//2):
                self.alpha_matrix[a, i + 1] = alpha_band[a, i]
                self.alpha_matrix[a, i + 1 + b] = np.conjugate(alpha_band[a, i])

    def _get_phases(self):
        Phi0 = 2 * np.pi / self._phaseStep
        self.phase_matrix = np.complex64(np.zeros((self._angleStep, self._phaseStep, self._phaseStep)))
        n_eff_phases = int((self._phaseStep + 1) / 2)  # number of effective phases terms
        for a in range(self._angleStep):
            self.phase_matrix[a, :, 0] = 1
            for i in range(self._phaseStep):
                for j in range(1, n_eff_phases):
                    self.phase_matrix[a, i, j] = np.exp(1j * (i + a * self._angleStep + 1) * (j + a * 2) * Phi0)
                    self.phase_matrix[a, i, j + n_eff_phases - 1] = np.exp(-1j * (i + a * self._angleStep + 1) * (j + a * 2) * Phi0)
                    print(i + a * self._angleStep + 1, (j + a * 2))

    def _ill_test(self, pstep, astep):
        return self.phase_matrix[astep, pstep, :] @ self.alpha_matrix[astep, :]



