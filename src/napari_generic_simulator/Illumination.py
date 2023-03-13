"""
Classes to calculate SIM illumination.
"""
__author__ = "Meizhu Liang @Imperial College London"

import torch

from .baseSIMulator import Base_simulator
import numpy as np
from numpy import cos, sin


class Illumination(Base_simulator):
    """A class to calculate illumination patterns of multiple beams."""
    def __init__(self):
        super().__init__()
        self._nsteps = int(self._phaseStep * self._angleStep)

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

    def jones_vectors(self):
        self.theta = np.arcsin(self.ill_NA / self.n)
        if (self.acc == 0) or (self.acc == 3):
            f_in = self.xp.transpose(self.xp.array(self.f_p))  # input field
            self.S = self.xp.zeros((self.npoints, self._n_beams, 3), dtype=self.xp.complex64)
            for i in range(self._n_beams):
                phi_S = i * self._beam_a
                self.S[:, i, :] = self.rotation(phi_S, self.theta) @ self.xp.array(
                    [[cos(phi_S), -sin(phi_S)], [sin(phi_S), cos(phi_S)], [0, 0]]) @ f_in
        else:
            f_in = torch.tensor(self.f_p, dtype=torch.float64, device=self._tdev)  # input field
            self.S = torch.zeros((self.npoints, self._n_beams, 3), dtype=torch.complex64, device=self._tdev)
            for i in range(self._n_beams):
                phi_S = i * self._beam_a
                self.S[:, i, :] = self.rotation(phi_S, self.theta) @ torch.tensor(
                    [[cos(phi_S), -sin(phi_S)], [sin(phi_S), cos(phi_S)], [0, 0]], device=self._tdev) @ f_in

    def _ill_test(self, x, y, pstep, astep):
        p = [0, pstep * 2 * np.pi / self._phaseStep, pstep * (-4) * np.pi / self._phaseStep]
        if (self.acc == 0) or (self.acc == 3):
            E = self.xp.zeros((self.npoints, self._n_beams, 3), dtype=self.xp.complex64)
            for i in range(self._n_beams):
                phi_E = i * self._beam_a + astep * 2 * np.pi / self._angleStep
                xyz = self.xp.transpose(self.xp.stack([x, y, self.xp.zeros(self.npoints)]))
                e = self.xp.exp(-1j * (xyz @ self.rotation(phi_E, self.theta) @ self.xp.array([0, 0, self.k0]) + p[i]))
                E[:, i, :] = self.xp.transpose(self.xp.array([e, ] * 3))
            F = self.xp.sum(self.S * E, axis=1, dtype=self.xp.complex64)
            ill = self.xp.sum(F * self.xp.conjugate(F), axis=1)  # the dot multiplication
            nor_ill = ill / self.xp.floor(self.xp.real(ill).max() + 0.5)  # normalised illumination
        else:
            E = torch.zeros((self.npoints, self._n_beams, 3), dtype=torch.complex64, device=self._tdev)
            for i in range(self._n_beams):
                phi_E = i * self._beam_a + astep * 2 * np.pi / self._angleStep
                xyz = torch.transpose(torch.stack([x, y, torch.zeros(self.npoints, device=self._tdev, dtype=torch.float64)]), 0, 1)
                e = torch.exp(-1j * (xyz @ self.rotation(phi_E, self.theta) @ torch.tensor([0, 0, self.k0], dtype=torch.float64, device=self._tdev) + p[i]))
                E[:, i, :] = torch.transpose(torch.stack((e, e, e)), 0, 1)
            F = torch.sum(self.S * E, axis=1, dtype=torch.complex64)
            ill = torch.sum(F * torch.conj(F), axis=1)  # the dot multiplication
            nor_ill = ill / torch.floor(torch.real(ill).max() + 0.5)  # normalised illumination
        return nor_ill


class ConIll(Illumination):
    def __init__(self):
        self._phaseStep = 3
        self._angleStep = 3
        self._n_beams = 2
        self._beam_a = 2 * np.pi / self._n_beams  # angle between each two beams
        self._nbands = 3
        super().__init__()


class HexIll(Illumination):
    def __init__(self):
        self._phaseStep = 7
        self._angleStep = 1
        self._n_beams = 3
        self._beam_a = 2 * np.pi / self._n_beams  # angle between each two beams
        self._nbands = 3
        super().__init__()


class RaHexIll(Illumination):
    def __init__(self):
        self._phaseStep = 5
        self._angleStep = 1
        self._n_beams = 3
        self._beam_a = np.pi / 2  # angle between beam1 and beam2, beam2 and beam3
        self._nbands = 3
        super().__init__()



