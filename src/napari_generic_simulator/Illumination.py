"""
Classes to calculate SIM illumination.
"""
__author__ = "Meizhu Liang @Imperial College London"

from .baseSIMulator import Base_simulator, import_torch
import numpy as np
from numpy import cos, sin


class Illumination(Base_simulator):
    """A class to calculate illumination patterns of multiple beams."""

    def __init__(self):
        super().__init__()
        self._nsteps = int(self._phaseStep * self._angleStep)
        # systematic errors, could be arbitrary
        if self.add_error:
            self.phase_error = np.reshape((2 * np.random.random(int(self._nsteps * self._n_beams)) - 1),
                                          (self._n_beams, self._angleStep, self._phaseStep))
            self.phase_error[0] = 0.0
            self.phase_error[:, :, 0] = 0.0
            # print(f'phase errors:{self.phase_error}')
            self.angle_error = np.reshape((2 * np.random.random(self._angleStep * self._n_beams) - 1),
                                          (self._n_beams, self._angleStep))
            self.angle_error[0] = 0.0
            self.angle_error[:, 0] = 0.0
            # print(f'angle errors:{self.angle_error}')
        else:
            self.phase_error = np.zeros((self._n_beams, self._angleStep, self._phaseStep))
            self.angle_error = np.zeros((self._n_beams, self._angleStep))

    def rotation(self, phi, theta):
        """Calculates the rotation matrix"""
        R = self.xp.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]) \
            @ self.xp.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]]) \
            @ self.xp.array([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
        return R

    def polarised_field(self, phi):
        if self.pol == 'a':
            f_p = self.xp.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)], [0, 0]]) @ self.xp.array([[0], [1]])
        elif self.pol == 'r':
            f_p = self.xp.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)], [0, 0]]) @ self.xp.array([[1], [0]])
        elif self.pol == 'c':
            f_p = self.xp.array([[1, 0], [0, 1], [0, 0]]) @ (self.xp.array([[1], [1j]]) / self.xp.sqrt(2))
        elif self.pol == 'h':
            f_p = self.xp.array([[1, 0], [0, 1], [0, 0]]) @ self.xp.array([[1], [0]])
        elif self.pol == 'v':
            f_p = self.xp.array([[1, 0], [0, 1], [0, 0]]) @ self.xp.array([[0], [1]])
        return f_p

    def jones_vectors(self, astep):
        self.theta = np.arcsin(self.ill_NA / self.n)
        self.S = self.xp.zeros((self.npoints, self._n_beams, 3), dtype=self.xp.complex64)
        for i in range(self._n_beams):
            phi_S = i * self._beam_a + astep * 2 * self.xp.pi / self._angleStep
            f_p = self.xp.array(self.polarised_field(phi_S))
            self.S[:, i, :] = self.xp.transpose(self.rotation(phi_S, self.theta) @ f_p)

    def _ill_obj(self, x, y, pstep, astep):
        """Illumination intensity applied on the object"""
        ill = self.xp.sum(self._ill_obj_vec(x, y, pstep, astep), axis=1)  # take real part and round to 15 decimals
        return ill

    def _ill_obj_vec(self, x, y, pstep, astep):
        """Vectorised illumination intensity applied on the object"""
        p = [0, pstep * 2 * np.pi / self._phaseStep, pstep * (-4) * np.pi / self._phaseStep]
        E = self.xp.zeros((self.npoints, self._n_beams, 3), dtype=self.xp.complex64)  # exponential terms of field
        for i in range(self._n_beams):
            phi_E = i * self._beam_a + astep * 2 * np.pi / self._angleStep + self.angle_error[i, astep]
            xyz = self.xp.transpose(self.xp.stack([x, y, self.xp.zeros(self.npoints)]))
            e = self.xp.exp(-1j * (xyz @ self.rotation(phi_E, self.theta) @ self.xp.array([0, 0, self.k0]) + p[i] +
                                   self.phase_error[i, astep, pstep]))
            E[:, i, :] = self.xp.transpose(self.xp.array([e, ] * 3))
        F = self.xp.sum(self.S * E, axis=1, dtype=self.xp.complex64)  # field of illumination
        # to calculate intensity: ill = F @ F_conjugate
        ill = (F * self.xp.conjugate(F)).real.round(15)  # take real part and round to 15 decimals
        return ill

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
