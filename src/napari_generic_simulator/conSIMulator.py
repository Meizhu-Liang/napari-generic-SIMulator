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
        self._n_steps = self._phaseStep * self._angleStep
        self._beam_c = np.array([[1, 0], [-1, 0]])  # beam components
        self._beam_a = 2 * np.pi / self._beam_c.shape[0]  # angle between each two beams
        super().__init__()

        # f_p: field components of different polarised beams
        # axial
        self.f_p = np.array([1, 0])
        # circular
        # self.f_p = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])
        # in-plane
        # self.f_p = np.array([0, 1])


    def _rotation(self, phi, theta):
        """rotation matrix for the field travelling in z, not for illumination patterns.
        phi is the azimuthal angle (0 - 2pi). theta is the polar angle (0 - pi)."""
        R = np.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]) \
            @ np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]]) \
            @ np.array([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
        return R

    def _ill(self):
        ill_intensity = np.zeros(self._n_steps, dtype=np.complex64)
        # xc, yc - Cartesian coordinate system
        xc = -1
        yc = 0
        for a in range(self._angleStep):
            angle = a * 2 * np.pi / self._angleStep
            # xr, yr - Cartesian coordinate system with rotation of axes
            xr = xc * np.cos(angle) + yc * np.sin(angle)
            yr = -xc * np.sin(angle) + yc * np.cos(angle)
            for p in range(self._phaseStep):
                _p1 = p * 2 * np.pi / self._phaseStep
                f_beams = np.zeros((self._beam_c.shape[0], 3), dtype=np.complex64)
                for i in range(self._beam_c.shape[0]):
                    # f_in: field of input beams
                    f_in = np.array([[cos(i * self._beam_a), -sin(i * self._beam_a)],
                                      [sin(i * self._beam_a), cos(i * self._beam_a)]]) @ self.f_p

                    # self.x, self.y= 1, 1

                    # f_beams: field of interfered beams
                    f_beams[i, :] = self._rotation(i * self._beam_a, theta=np.pi / 2) @ np.array([[1, 0], [0, 1], [0, 0]]) @ f_in * np.exp(
                        -1j * (self.ph * np.array([[cos(i * self._beam_a), -sin(i * self._beam_a)],
                                      [sin(i * self._beam_a), cos(i * self._beam_a)]]) @ np.array([xr, yr]) @ np.array([self.x, self.y]) - i * _p1))
                    if not hasattr(self, 'print'):
                        print((xr, yr))
                f_total = np.sum(f_beams, axis=0)
                ill_intensity[p + self._phaseStep * a] = f_total @ np.conj(f_total)
                # if not hasattr(self, 'print'):
                    # print('================')
                    # print(f_total)
                    # print(p + self._phaseStep * a)
                    # print(xr * self.x, yr * self.y, p * i)
                #     # print(self._beam_c[i] @ np.array([xr, yr]))
                #     print(ill_intensity.max())
        if not hasattr(self, 'print'):
            print(ill_intensity)
        self.print = True
        return ill_intensity


class Illumination(Base_simulator):
    """
    A class to calculate illumination patterns of multiple beams.
    """

    def __init__(self):
        self._phaseStep = 3
        self._angleStep = 3
        self._n_beams = 2
        self._n_steps = self._phaseStep * self._angleStep
        self._beam_c = np.array([[1, 0], [-1, 0]])  # beam components
        self._beam_a = 2 * np.pi / self._beam_c.shape[0]  # angle between each two beams
        super().__init__()

        # f_p: field components of different polarised beams
        # axial
        self.f_p = np.array([1, 0])
        # circular
        # self.f_p = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])
        # in-plane
        # self.f_p = np.array([0, 1])


    def _rotation(self, phi, theta):
        """rotation matrix for the field travelling in z, not for illumination patterns.
        phi is the azimuthal angle (0 - 2pi). theta is the polar angle (0 - pi)."""
        R = np.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]) \
            @ np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]]) \
            @ np.array([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
        return R

    def _ill(self):
        ill_intensity = np.zeros(self._n_steps, dtype=np.complex64)
        # xc, yc - Cartesian coordinate system
        xc = -1
        yc = 0
        for a in range(self._angleStep):
            angle = a * 2 * np.pi / self._angleStep
            # xr, yr - Cartesian coordinate system with rotation of axes
            xr = xc * np.cos(angle) + yc * np.sin(angle)
            yr = -xc * np.sin(angle) + yc * np.cos(angle)
            for p in range(self._phaseStep):
                _p1 = p * 2 * np.pi / self._phaseStep
                f_beams = np.zeros((self._beam_c.shape[0], 3), dtype=np.complex64)
                for i in range(self._beam_c.shape[0]):
                    # f_in: field of input beams
                    f_in = np.array([[cos(i * self._beam_a), -sin(i * self._beam_a)],
                                      [sin(i * self._beam_a), cos(i * self._beam_a)]]) @ self.f_p

                    # self.x, self.y= 1, 1

                    # f_beams: field of interfered beams
                    f_beams[i, :] = self._rotation(i * self._beam_a, theta=np.pi / 2) @ np.array([[1, 0], [0, 1], [0, 0]]) @ self.f_p * np.exp(
                        -1j * (self.ph * np.array([[cos(i * self._beam_a), -sin(i * self._beam_a)],
                                      [sin(i * self._beam_a), cos(i * self._beam_a)]]) @ np.array([xr, yr]) @ np.array([self.x, self.y]) - i * _p1))
                    if not hasattr(self, 'print'):
                        print((xr, yr))
                f_total = np.sum(f_beams, axis=0)
                ill_intensity[p + self._phaseStep * a] = f_total @ np.conj(f_total)
                # if not hasattr(self, 'print'):
                    # print('================')
                    # print(f_total)
                    # print(p + self._phaseStep * a)
                    # print(xr * self.x, yr * self.y, p * i)
                #     # print(self._beam_c[i] @ np.array([xr, yr]))
                #     print(ill_intensity.max())
        if not hasattr(self, 'print'):
            print(ill_intensity)
        self.print = True
        return ill_intensity

    def _get_illumination(self, astep:int, pstep:int):
        phases = np.zeros(self._phaseStep)
        theta = np.pi / 2  # half angle (polar) of the detective beams
        k = np.array([1,0,0], [-1,0,0])
        # S_beams: Jones vector; E_beams: exponential term; a beam could be expressed as S_beams @ E_beams
        S_beams, E_beams = np.zeros((self._n_beams, self._phaseStep)), np.zeros((self._n_beams, self._phaseStep))
        alpha = np.zeros((self._phaseStep, self._n_beams, self._phaseStep))
        for i in self._n_beams:
            phi = i * self._beam_a
            R = np.array([[cos(phi),-sin(phi),0], [sin(phi),cos(phi),0], [0,0,1]]) \
                @ np.array([[cos(theta),0,-sin(theta)], [0,1,0], [sin(theta),0,cos(theta)]]) \
                @ np.array([[cos(phi),sin(phi),0], [-sin(phi),cos(phi),0], [0,0,1]])
            S_beams[i, :] = R @ np.array([[1, 0], [0, 1], [0, 0]]) @ self.f_p
            E_beams[i, :] = np.exp(-1j *(np.array([self.x, self.y, 0]) @ R @ k[i]))
            for p in self._phaseStep:
                for i in self._n_beams:
                    alpha[p, i, :] = S_beams[i] * np.conjugate(S_beams[i]) * E_beams * np.conjugate(E_beams)
            for p in self._phaseStep:
                phases[p] = np.exp(1j * (p * 2 * np.pi / self._phaseStep))
            ill = phases[pstep]

