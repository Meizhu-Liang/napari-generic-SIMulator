"""
Classes to calculate SIM illumination.
"""
__author__ = "Meizhu Liang @Imperial College London"

import torch

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

    def rotation(self, phi, theta):
        """Calculates the rotation matrix"""
        if (self.acc == 0) or (self.acc == 3):
            R = self.xp.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]) \
                @ self.xp.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]]) \
                @ self.xp.array([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
        else:
            R = torch.tensor([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]], device=self._tdev) \
                @ torch.tensor([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]],
                               device=self._tdev) \
                @ torch.tensor([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]], device=self._tdev)
        return R

    def _get_alpha_constants(self):
        self.theta = np.arcsin(self.ill_NA / self.n)
        # S_beams: Jones vector; E_beams: exponential term; a beam could be expressed as S_beams @ E_beams
        if (self.acc == 0) or (self.acc == 3):
            self.S_beams = self.xp.zeros((self._angleStep, self._n_beams, 3), dtype=self.xp.complex64)
            self.alpha_matrix = self.xp.zeros((self.npoints, self._angleStep, self._phaseStep), dtype=self.xp.complex64)
            con = self.xp.zeros((self._angleStep, self._n_beams), dtype=self.xp.complex64)  # constant alpha values
            f_in = self.xp.array(self.f_p)  # input field

            for a in range(self._angleStep):
                for i in range(self._n_beams):
                    phi = i * self._beam_a + a * 2 * np.pi / self._angleStep
                    # rotation matrix for the field travelling in z, not for illumination patterns.
                    # phi is the azimuthal angle (0 - 2pi). theta is the polar angle (0 - pi).
                    self.S_beams[a, i, :] = self.rotation(phi, self.theta) @ self.xp.array(
                        [[cos(phi), -sin(phi)], [sin(phi), cos(phi)], [0, 0]]) @ f_in
                    con[a, i] = self.S_beams[a, i] @ self.xp.conjugate(self.S_beams[a, i])
                # constant alpha values
                self.alpha_matrix[:, a, 0] = self.xp.sum(con[a])
        else:
            self.S_beams = torch.zeros(self._angleStep, self._n_beams, 3, dtype=torch.complex64, device=self._tdev)
            self.alpha_matrix = torch.zeros(self.npoints, self._angleStep, self._phaseStep, dtype=torch.complex64,
                                            device=self._tdev)
            con = torch.zeros(self._angleStep, self._n_beams, dtype=torch.complex64,
                              device=self._tdev)  # constant alpha values
            f_in = torch.tensor(self.f_p, device=self._tdev, dtype=torch.float64)  # input field

            for a in range(self._angleStep):
                for i in range(self._n_beams):
                    phi = i * self._beam_a + a * 2 * np.pi / self._angleStep
                    # rotation matrix for the field travelling in z, not for illumination patterns.
                    # phi is the azimuthal angle (0 - 2pi). theta is the polar angle (0 - pi).
                    self.S_beams[a, i, :] = self.rotation(phi, self.theta) @ torch.tensor(
                        [[cos(phi), -sin(phi)], [sin(phi), cos(phi)], [0, 0]], device=self._tdev) @ f_in
                    con[a, i] = self.S_beams[a, i] @ torch.conj(self.S_beams[a, i])
                # constant alpha values
                self.alpha_matrix[:, a, 0] = torch.sum(con[a])

    def _get_alpha(self, x, y, astep):
        if (self.acc == 0) or (self.acc == 3):
            self.E_beams = self.xp.zeros((self.npoints, self._angleStep, self._n_beams), dtype=self.xp.complex64)
            self.alpha_band = self.xp.zeros((self.npoints, self._angleStep, self._nbands), dtype=self.xp.complex64)
            xyz = self.xp.transpose(self.xp.stack([x, y, self.xp.zeros(self.npoints)]))

            for i in range(self._n_beams):
                phi = i * self._beam_a + astep * 2 * np.pi / self._angleStep
                self.E_beams[:, astep, i] = self.xp.exp(
                    -1j * (xyz @ self.rotation(phi, self.theta) @
                           self.xp.array([0, 0, self.k0])))
            b = 0
            for i in range(self._n_beams):
                for j in range(int(self._n_beams - i - 1)):
                    self.alpha_band[:, astep, b] = self.S_beams[astep, i] @ self.xp.conjugate(
                        self.S_beams[astep, i + j + 1]) * self.E_beams[:, astep, i] * self.xp.conjugate(
                        self.E_beams[:, astep, i + j + 1])
                    b += 1
            for i in range((self._phaseStep - 1) // 2):
                self.alpha_matrix[:, astep, i + 1] = self.alpha_band[:, astep, i]
                self.alpha_matrix[:, astep, i + 1 + b] = self.xp.conjugate(self.alpha_band[:, astep, i])
        else:
            self.E_beams = torch.zeros(self.npoints, self._angleStep, self._n_beams, dtype=torch.complex64,
                                       device=self._tdev)
            self.alpha_band = torch.zeros(self.npoints, self._angleStep, self._nbands, dtype=torch.complex64,
                                          device=self._tdev)

            xyz = torch.transpose(
                torch.stack([x, y, torch.zeros(self.npoints, device=self._tdev, dtype=torch.float64)]), 0, 1)

            for i in range(self._n_beams):
                phi = i * self._beam_a + astep * 2 * np.pi / self._angleStep
                self.E_beams[:, astep, i] = torch.exp(
                    -1j * (torch.matmul(torch.matmul(xyz, self.rotation(phi, self.theta)),
                                        torch.tensor([0, 0, self.k0], device=self._tdev, dtype=torch.float64))))
            b = 0
            for i in range(self._n_beams):
                for j in range(int(self._n_beams - i - 1)):
                    self.alpha_band[:, astep, b] = self.S_beams[astep, i] @ torch.conj(
                        self.S_beams[astep, i + j + 1]) * self.E_beams[:, astep, i] * torch.conj(
                        self.E_beams[:, astep, i + j + 1])
                    b += 1

            for i in range((self._phaseStep - 1) // 2):
                self.alpha_matrix[:, astep, i + 1] = self.alpha_band[:, astep, i]
                self.alpha_matrix[:, astep, i + 1 + b] = torch.conj(self.alpha_band[:, astep, i])
        return self.alpha_matrix

    def _ill_test(self, x, y, pstep, astep):
        return self._get_alpha(x, y, astep)[:, astep, :] @ self.phase_matrix[pstep]


class ConIll(Illumination):
    def __init__(self):
        self._phaseStep = 3
        self._angleStep = 3
        self._n_beams = 2
        super().__init__()

        def _get_phases():
            # phase matrix
            Phi0 = 2 * np.pi / self._phaseStep
            self.phase_matrix = self.xp.ones((self._phaseStep, int(self._nbands / self._angleStep * 2 + 1)),
                                             dtype=self.xp.complex64)
            for p in range(self._phaseStep):
                for i in range(int(self._nbands / self._angleStep)):
                    self.phase_matrix[p, i + 1] = self.xp.exp(1j * (p * (i + 1) * Phi0))
                    self.phase_matrix[p, int(i + 1 + self._nbands / self._angleStep)] = self.xp.exp(
                        -1j * (p * (i + 1) * Phi0))
        _get_phases()


class HexIll(Illumination):
    def __init__(self):
        self._phaseStep = 7
        self._angleStep = 1
        self._n_beams = 3
        super().__init__()

        def _get_phases():
            # phase matrix
            Phi0 = 2 * np.pi / self._phaseStep
            self.phase_matrix = self.xp.ones((self._phaseStep, int(self._nbands / self._angleStep * 2 + 1)),
                                             dtype=self.xp.complex64)
            for p in range(self._phaseStep):
                for i in range(int(self._nbands / self._angleStep)):
                    self.phase_matrix[p, i + 1] = self.xp.exp(1j * (p * (i + 1) * Phi0))
                    self.phase_matrix[p, int(i + 1 + self._nbands / self._angleStep)] = self.xp.exp(
                        -1j * (p * (i + 1) * Phi0))
        _get_phases()

