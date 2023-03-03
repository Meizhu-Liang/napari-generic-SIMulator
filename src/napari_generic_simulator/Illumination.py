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
        self._phaseStep = 7
        self._angleStep = 1
        self._n_beams = 3
        self._nsteps = int(self._phaseStep * self._angleStep)
        self._nbands = int((self._nsteps - self._angleStep) / 2)
        self._beam_a = 2 * pi / self._n_beams  # angle between each two beams
        super().__init__()

        # f_p: field components of different polarised beams
        # axial
        self.f_p = np.array([1, 0])
        # circular
        # self.f_p = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])
        # in-plane
        # self.f_p = np.array([0, 1])

    def rotation_matrix(self, phi, theta=pi / 2):
        R = np.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]) \
            @ np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]]) \
            @ np.array([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
