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
        self.xc = 1
        self.yc = 1
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
        self._n_steps = self._phaseStep * self._angleStep
        self._beam_c = np.array([[1, 0], [-1, 0]])  # beam components
        self._beam_a = 2 * np.pi / self._beam_c.shape[0]  # angle between each two beams
        super().__init__()

        # f_p: field components of different polarised beams
        self.f_p = np.array([1, 0])
        # self.f_p = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])
        # self.f_p = np.array([0, 1])


    def _rotation(self, phi, theta):
        """rotation matrix for the field travelling in z, not for illumination patterns.
        phi is the azimuthal angle (0 - 2pi). theta is the polar angle (0 - pi)."""
        R = np.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]) \
            @ np.array([[cos(theta), 0], [0, 1], [sin(theta), 0]]) \
            @ np.array([[cos(phi), sin(phi)], [-sin(phi), cos(phi)]])
        return R

    def _ill(self):
        ill_intensity = np.zeros(self._n_steps)
        # xc, yc - Cartesian coordinate system
        xc = -1
        yc = 0
        for a in range(self._angleStep):
            angle = a * 2 * np.pi / self._angleStep
            # xr, yr - Cartesian coordinate system with rotation of axes
            xr = xc * np.cos(angle) + yc * np.sin(angle)
            yr = xc * np.sin(angle) + yc * np.cos(angle)
            for p in range(self._phaseStep):
                _p1 = p * 2 * np.pi / self._phaseStep
                f_beams = 0
                for i in range(self._beam_c.shape[0]):
                    # f_in: field of input beams
                    f_in = np.array([[cos(i * self._beam_a), -sin(i * self._beam_a)],
                                      [sin(i * self._beam_a), cos(i * self._beam_a)]]) @ self.f_p

                    self.x, self.y = 1, 1

                    # f_beams: field of interfered beams
                    f_beams += self._rotation(i * self._beam_a, theta=np.pi / 2) @ f_in * np.exp(-1j * (self.ph * np.array(self._beam_c[i]) @ np.array([xr * self.x, yr * self.y]) - i * _p1))
                ill_intensity[p + self._phaseStep * a] = np.dot(f_beams, np.conj(f_beams))
                if not hasattr(self, 'print'):
                    print(xr, yr, p * i)
                #     # print(self._beam_c[i] @ np.array([xr, yr]))
                #     print(ill_intensity.max())
        if not hasattr(self, 'print'):
            print(ill_intensity)
        self.print = True
        return ill_intensity
