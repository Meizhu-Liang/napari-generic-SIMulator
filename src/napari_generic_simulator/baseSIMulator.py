"""
The parent class to simulate raw data for SIM.
Some calculations are adapted by the work by Mark Neil @Imperial College London
"""
__author__ = "Meizhu Liang @Imperial College London"

import numpy as np
import time
import tifffile

try:
    import cupy as cp

    print('cupy imported')
    import_cp = True
except:
    import_cp = False

try:
    import torch
    print('torch imported')
    import_torch = True
    if torch.has_cuda:
        torch_GPU = True
    else:
        torch_GPU = False
except:
    import_torch = False
    torch_GPU = False

class Base_simulator:
    points = None
    npoints = 0
    xp = np
    pol = None  # polarisation
    acc = None  # acceleration
    psf_calc = None
    _tdev = None
    N = 512  # points to use in FFT
    pixel_size = 5.5  # camera pixel size
    magnification = 60  # objective magnification
    ill_NA = 0.75  # numerical aperture at illumination beams
    det_NA = 1.1  # numerical aperture at sample
    n = 1.33  # refractive index at sample
    ill_wavelength = 520  # illumination wavelength in nm
    det_wavelength = 570  # detection wavelength in nm
    zrange = 7.0  # distance either side of focus to calculate, in microns, could be arbitrary
    dz = 0.4  # step size in axial direction of PSF
    fwhmz = 3.0  # FWHM of light sheet in z
    random_seed = 123
    drift = 0
    defocus = 0  # de-focus aberration in um
    spherical = 0
    sph_abb = 0
    add_error = True

    def initialise(self):
        if self.acc == 3:
            self.xp = cp
        np.random.seed(self.random_seed)
        # self.seed(1234)  # set random number generator seed
        self.ill_wavelength = self.ill_wavelength * 1e-3
        self.det_wavelength = self.det_wavelength * 1e-3
        self.sigmaz = self.fwhmz / 2.355
        self.dx = self.pixel_size / self.magnification  # Sampling in lateral plane at the sample in um
        self.dxn = self.det_wavelength / (4 * self.det_NA)  # 2 * Nyquist frequency in x and y.
        self.Nn = int(np.ceil(self.N * self.dx / self.dxn / 2) * 2)  # Number of points at Nyquist sampling, even number
        self.dxn = self.N * self.dx / self.Nn  # correct spacing
        self.res = self.det_wavelength / (2 * self.det_NA)
        oversampling = self.res / self.dxn  # factor by which pupil plane oversamples the coherent psf data
        self.dk = oversampling / (self.Nn / 2)  # Pupil plane sampling
        self.k0 = 2 * np.pi * self.n / self.det_wavelength
        self.kx, self.ky = self.xp.meshgrid(self.xp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn),
                                       self.xp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn))
        self.kr = np.sqrt(self.kx ** 2 + self.ky ** 2)  # Raw pupil function, pupil defined over circle of radius 1.
        self.krmax = self.det_NA * self.k0 / self.n
        self.kr2 = self.kx ** 2 + self.ky ** 2
        self.spherical = self.sph_abb * np.sqrt(5) * (6 * (self.kr ** 4 - self.kr ** 2) + 1)
        self.csum = sum(sum((self.kr < 1)))  # normalise by csum so peak intensity is 1

        self.alpha = np.arcsin(self.det_NA / self.n)
        # Nyquist sampling in z, reduce by 10 % to account for gaussian light sheet
        self.dzn = 0.8 * self.det_wavelength / (2 * self.n * (1 - np.cos(self.alpha)))
        self.Nz = int(2 * np.ceil(self.zrange / self.dz))
        self.dz = 2 * self.zrange / self.Nz
        self.Nzn = int(2 * np.ceil(self.zrange / self.dzn))
        self.dzn = 2 * self.zrange / self.Nzn
        if self.Nz < self.Nzn:
            self.Nz = self.Nzn
            self.dz = self.dzn
        else:
            self.Nzn = self.Nz
            self.dzn = self.dz
        if (self.acc == 1) | (self.acc == 2):
            self._tdev = torch.device('cuda' if self.acc == 2 else 'cpu')

    # def point_cloud_fil(self):
    #     """
    #     Generates a point-cloud as the object in the imaging system.
    #     """
    #
    #     L = 5  # full length of each filament in um
    #     dL = 0.05  # target length of each short bit to form a filament
    #
    #     for i in range(self.nSamples):
    #         alpha = 2 * np.pi * np.random.rand()
    #         beta = np.pi / 4 * (1 - 2 * np.random.rand())
    #
    #         # centres of filaments
    #         xS = -L / 2 + L * np.random.rand()
    #         yS = -L / 2 + L * np.random.rand()
    #         zS = 0
    #
    #         # starting points of filaments
    #         x1 = xS + (L / 2 * np.cos(alpha) * np.cos(beta))
    #         y1 = yS + (L / 2 * np.sin(alpha) * np.cos(beta))
    #         z1 = zS + (L / 2 * np.sin(beta))
    #
    #         # ending points of filaments
    #         x2 = xS - (L / 2 * np.cos(alpha) * np.cos(beta))
    #         y2 = yS - (L / 2 * np.sin(alpha) * np.cos(beta))
    #         z2 = zS - (L / 2 * np.sin(beta))
    #
    #         px = 2 * np.pi * np.random.rand()
    #         fx = 2 * np.pi * 0.5 * np.random.rand()
    #         ax = L / 10 * np.random.rand()
    #         py = 2 * np.pi * np.random.rand()
    #         fy = 2 * np.pi * 1.0 * np.random.rand()
    #         ay = L / 10 * np.random.rand()
    #         pz = 2 * np.pi * np.random.rand()
    #         fz = 2 * np.pi * 1.0 * np.random.rand()
    #         az = L / 10 * np.random.rand()
    #
    #         # adding some perturbation
    #         l = np.arange(0, 1, dL / L)
    #         x = x1 + (x2 - x1) * l + ax * np.cos(fx * l + px)
    #         y = y1 + (y2 - y1) * l + ay * np.cos(fy * l + py)
    #         z = z1 + (z2 - z1) * l + az * np.cos(fz * l + pz)
    #
    #         xd = x[1:] - x[:-1]
    #         yd = y[1:] - y[:-1]
    #         zd = z[1:] - z[:-1]
    #
    #         # steps = np.sqrt(xd ** 2 + yd ** 2 + zd ** 2)
    #         # print(f'length = {np.sum(steps):.2f} um, average = {1000 * np.mean(steps) :.2f} nm, stdev = {1000 * np.std(steps) :.2f} nm')
    #         if i == 0:
    #             self.points = np.single(np.dstack([x, y, z]).squeeze())
    #         else:
    #             self.points = np.concatenate((self.points, np.single(np.dstack([x, y, z]).squeeze())))
    #     self.npoints = self.points.shape[0]

    # def point_cloud(self):
    #     """
    #     Generates a point-cloud as the object in the imaging system.
    #     """
    #     rad = 5  # radius of sphere of points
    #     # Multiply the points several times to get the enough number
    #     pointsxn = (2 * np.random.rand(self.nSamples * 3, 3) - 1) * [rad, rad, rad]
    #
    #     pointsxnr = np.sum(pointsxn * pointsxn, axis=1)  # multiple times the points
    #     points_sphere = pointsxn[pointsxnr < (rad ** 2), :]  # simulate spheres from cubes
    #     self.points = np.single(points_sphere[(range(self.nSamples)), :])
    #     self.points[:, 2] = self.points[:, 2] / 2  # to make the point cloud for OTF a ellipsoid rather than a sphere
    #     self.npoints = self.nSamples

    def phase_tilts(self):
        """Generates phase tilts in frequency space"""
        xyrange = self.Nn / 2 * self.dxn
        dkxy = np.pi / xyrange
        dkz = np.pi / self.zrange
        if self.acc == 0:
            self.kxy = np.arange(-self.Nn / 2 * dkxy, (self.Nn / 2) * dkxy, dkxy, dtype=np.single)
            self.kz = np.arange(-self.Nzn / 2 * dkz, (self.Nzn / 2) * dkz, dkz, dtype=np.single)
        elif self.acc == 3:
            self.kxy = cp.arange(-self.Nn / 2 * dkxy, (self.Nn / 2) * dkxy, dkxy, dtype=cp.single)
            self.kz = cp.arange(-self.Nzn / 2 * dkz, (self.Nzn / 2) * dkz, dkz, dtype=cp.single)
        else:
            self.kxy = torch.arange(-self.Nn / 2 * dkxy, (self.Nn / 2) * dkxy, dkxy,
                                    dtype=torch.float32, device=self._tdev)
            self.kz = torch.arange(-self.Nzn / 2 * dkz, (self.Nzn / 2) * dkz, dkz,
                                   dtype=torch.float32, device=self._tdev)

        if self.acc == 0:
            self.phasetilts = np.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
        elif self.acc == 3:
            self.phasetilts = cp.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=cp.complex64)
        else:
            self.phasetilts = torch.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=torch.complex64,
                                          device=self._tdev)

        start_time = time.time()
        itcount = 0
        total_its = self._angleStep * self._phaseStep * self.npoints
        lastProg = 0
        self.ph = 4 * np.pi * self.ill_NA / self.ill_wavelength
        for astep in range(self._angleStep):
            for pstep in range(self._phaseStep):
                self.points += self.drift * np.random.standard_normal(3) / 1000
                self.points[:, 0] += self.xdrift / 1000
                self.points[:, 2] += self.zdrift / 1000
                isteps = pstep + self._phaseStep * astep  # index of the steps
                for i in range(self.npoints):
                    prog = (100 * itcount) // total_its
                    if prog > lastProg + 9:
                        lastProg = prog
                        yield f'Phase tilts calculation: {prog:.1f}% done'
                    itcount += 1
                    self.x = self.points[i, 0]
                    self.y = self.points[i, 1]
                    z = self.points[i, 2]
                    # get illumination from the child class
                    if self.pol == 'axial':
                        ill = self._illAx(pstep, astep)
                    elif self.pol == 'circular':
                        ill = self._illCi(pstep, astep)
                    else:
                        ill = self._illIp(pstep, astep)
                    if self.acc == 0:
                        px = np.exp(1j * np.single(self.x * self.kxy))
                        py = np.exp(1j * np.single(self.y * self.kxy))[:, np.newaxis]
                        pz = (np.exp(1j * np.single(z * self.kz)) * ill)[:, np.newaxis, np.newaxis]
                        self.phasetilts[isteps, :, :, :] += (px * py) * pz
                    elif self.acc == 3:
                        px = cp.exp(1j * self.x * self.kxy)
                        py = cp.exp(1j * self.y * self.kxy)[:, cp.newaxis]
                        pz = (cp.exp(1j * z * self.kz) * ill)[:, cp.newaxis, cp.newaxis]
                        self.phasetilts[isteps, :, :, :] += (px * py) * pz
                    else:
                        px = torch.exp(1j * self.x * self.kxy)
                        py = torch.exp(1j * self.y * self.kxy)
                        pz = torch.exp(1j * z * self.kz) * ill
                        self.phasetilts[isteps, :, :, :] += (px * py[..., None]) * pz[..., None, None]
        self.elapsed_time = time.time() - start_time
        yield f'Phase tilts calculation time:  {self.elapsed_time:3f}s'

    def get_vector_psf(self):
        # use krmax to define the pupil function
        kx = self.krmax * (self.kx + 1e-7)  # need to add small offset to avoid division by zero
        ky = self.krmax * (self.ky + 1e-7)
        kr2 = (kx ** 2 + ky ** 2)  # square kr
        e_in = 1.0 * (kr2 < self.krmax ** 2)
        kz = np.sqrt((self.k0 ** 2 - kr2) + 0j)

        # Calculating psf
        nz = 0
        psf = self.xp.zeros((self.Nzn, self.Nn, self.Nn))

        # calculate intensity of random arrangement of dipoles excited by a given polarisation s
        # p are the vertices of an dodecahedron
        p0 = self.xp.reshape(self.xp.array([0, 1, 0]), (1, 3))
        p1 = self.xp.reshape(self.xp.array([-0.666666, 0., 0.745353,
                                  0.666666, 0., -0.745353,
                                  -0.127322, -0.93417, 0.333332,
                                  -0.127322, 0.93417, 0.333332,
                                  0.745355, -0.577349, -0.333332,
                                  0.745355, 0.577349, -0.333332,
                                  0.333332, -0.577349, 0.745355,
                                  0.333332, 0.577349, 0.745355,
                                  -0.872676, -0.356821, -0.333334,
                                  -0.872676, 0.356821, -0.333334,
                                  0.872676, -0.356821, 0.333334,
                                  0.872676, 0.356821, 0.333334,
                                  1.46634 * 1e-6, 0., -0.999998,
                                  -0.745355, -0.577349, 0.333332,
                                  -0.745355, 0.577349, 0.333332,
                                  -1.46634 * 1e-6, 0., 0.999998,
                                  -0.333332, -0.577349, -0.745355,
                                  -0.333332, 0.577349, -0.745355,
                                  0.127322, -0.93417, -0.333332,
                                  0.127322, 0.93417, -0.333332]), (20, 3))
        # p2 are the vertices of the same icosahedron in a different orientation
        p2 = self.xp.reshape(self.xp.array([-1.37638, 0., 0.262866,
                                  1.37638, 0., -0.262866,
                                  -0.425325, -1.30902, 0.262866,
                                  -0.425325, 1.30902, 0.262866,
                                  1.11352, -0.809017, 0.262866,
                                  1.11352, 0.809017, 0.262866,
                                  -0.262866, -0.809017, 1.11352,
                                  -0.262866, 0.809017, 1.11352,
                                  -0.688191, -0.5, -1.11352,
                                  -0.688191, 0.5, -1.11352,
                                  0.688191, -0.5, 1.11352,
                                  0.688191, 0.5, 1.11352,
                                  0.850651, 0., -1.11352,
                                  -1.11352, -0.809017, -0.262866,
                                  -1.11352, 0.809017, -0.262866,
                                  -0.850651, 0., 1.11352,
                                  0.262866, -0.809017, -1.11352,
                                  0.262866, 0.809017, -1.11352,
                                  0.425325, -1.30902, -0.262866,
                                  0.425325, 1.30902, -0.262866]), (20, 3))
        p2 = p2 / self.xp.linalg.norm(p2[0, :])

        p = p2
        s1 = self.xp.array([1, 0, 0])  # x polarised illumination orientation
        excitation1 = (s1 @ p.T) ** 2
        s2 = self.xp.array([0, 1, 0])  # y polarised illumination orientation
        excitation2 = (s2 @ p.T) ** 2
        s3 = self.xp.array([0, 0, 1])  # z polarised illumination orientation
        excitation3 = (s3 @ p.T) ** 2

        fx1 = self.xp.sqrt(self.k0 / kz) * (self.k0 * ky ** 2 + kx ** 2 * kz) / (self.k0 * kr2) * e_in
        fy1 = self.xp.sqrt(self.k0 / kz) * kx * ky * (kz - self.k0) / (self.k0 * kr2) * e_in
        fx2 = self.xp.sqrt(self.k0 / kz) * kx * ky * (kz - self.k0) / (self.k0 * kr2) * e_in
        fy2 = self.xp.sqrt(self.k0 / kz) * (self.k0 * kx ** 2 + ky ** 2 * kz) / (self.k0 * kr2) * e_in
        fx3 = self.xp.sqrt(self.k0 / kz) * kx / self.k0 * e_in
        fy3 = self.xp.sqrt(self.k0 / kz) * ky / self.k0 * e_in

        for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
            Exx = self.xp.fft.fftshift(self.xp.fft.ifft2(fx1 * np.exp(1j * z * kz)))  # x-polarised field at camera for x-oriented dipole
            Exy = self.xp.fft.fftshift(self.xp.fft.ifft2(fy1 * np.exp(1j * z * kz)))  # y-polarised field at camera for x-oriented dipole
            Eyx = self.xp.fft.fftshift(self.xp.fft.ifft2(fx2 * np.exp(1j * z * kz)))  # x-polarised field at camera for y-oriented dipole
            Eyy = self.xp.fft.fftshift(self.xp.fft.ifft2(fy2 * np.exp(1j * z * kz)))  # y-polarised field at camera for y-oriented dipole
            Ezx = self.xp.fft.fftshift(self.xp.fft.ifft2(fx3 * np.exp(1j * z * kz)))  # x-polarised field at camera for z-oriented dipole
            Ezy = self.xp.fft.fftshift(self.xp.fft.ifft2(fy3 * np.exp(1j * z * kz)))  # y-polarised field at camera for z-oriented dipole
            intensityx = self.xp.zeros((self.Nn, self.Nn))
            intensityy = self.xp.zeros((self.Nn, self.Nn))
            intensityz = self.xp.zeros((self.Nn, self.Nn))
            for i in np.arange(p.shape[0]):
                intensityx = intensityx + excitation1[i] * (abs(p[i, 0] * Exx + p[i, 1] * Eyx + p[i, 2] * Ezx) ** 2
                                                            + abs(p[i, 0] * Exy + p[i, 1] * Eyy + p[i, 2] * Ezy) ** 2)
                intensityy = intensityy + excitation2[i] * (abs(p[i, 0] * Exx + p[i, 1] * Eyx + p[i, 2] * Ezx) ** 2 +
                                                            abs(p[i, 0] * Exy + p[i, 1] * Eyy + p[i, 2] * Ezy) ** 2)
                intensityz = intensityz + excitation3[i] * (abs(p[i, 0] * Exx + p[i, 1] * Eyx + p[i, 2] * Ezx) ** 2 +
                                                           abs(p[i, 0] * Exy + p[i, 1] * Eyy + p[i, 2] * Ezy) ** 2)
            if self.pol == 'axial':
                intensity = intensityz  # for axially polarised illumination
            elif self.pol == 'circular':
                # dipoles that re-orient between excitation and emmission and maybe for circular polarised illumination
                intensity = (intensityx + intensityy + intensityz) / 3
            else:
                intensity = intensityx + intensityy  # for in plane illumination
            psf[nz, :, :] = intensity * self.xp.exp(-z ** 2 / 2 / self.sigmaz ** 2)

            nz = nz + 1
        psf = psf * self.Nn ** 2 / self.xp.sum(e_in) * self.Nz / self.Nzn
        return psf

    def get_scalar_psf(self):
        # use krmax to define the pupil function
        kx = self.krmax * self.kx
        ky = self.krmax * self.ky
        kr2 = (kx ** 2 + ky ** 2)  # square kr
        # 2 * np.pi * self.n / self.wavelength
        kz = self.xp.sqrt((self.k0 ** 2 - kr2) + 0j)
        nz = 0
        psf = self.xp.zeros((self.Nzn, self.Nn, self.Nn))
        pupil = self.kr < 1
        for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
            c = (np.exp(
                1j * (z * self.n * 2 * np.pi / self.det_wavelength *
                      np.sqrt(1 - (self.kr * pupil) ** 2 * self.det_NA ** 2 / self.n ** 2)))) * pupil
            psf[nz, :, :] = abs(np.fft.fftshift(np.fft.ifft2(c))) ** 2 * np.exp(-z ** 2 / 2 / self.sigmaz ** 2)
            nz = nz + 1
        # Normalised so power in resampled psf(see later on) is unity in focal plane
        psf = psf * self.Nn ** 2 / self.xp.sum(pupil) * self.Nz / self.Nzn
        return psf

    # def raw_image_stack(self):
    #     # Calculates point cloud, phase tilts, 3d psf and otf before the image stack.
    #     self.initialise()
    #     self.drift = 0.0  # no random walk using this method
    #     self.point_cloud()
    #     yield "Point cloud calculated"
    #
    #     self._nsteps = self._phaseStep * self._angleStep
    #     for msg in self.phase_tilts():
    #         yield msg
    #     if self.psf_calc == 'vector':
    #         psf = self.get_vector_psf()
    #     else:
    #         psf = self.get_scalar_psf()
    #     self.psf_z0 = psf[int(self.Nzn / 2 + 5), :, :]  # psf at z=0
    #     if self.acc == 3:
    #         self.psf_z0 = cp.asnumpy(self.psf_z0)
    #     yield "psf calculated"
    #
    #     # Calculating 3d otf
    #     otf = self.xp.fft.fftn(psf)
    #     aotf = abs(self.xp.fft.fftshift(otf))  # absolute otf
    #     if self.acc == 3:
    #         aotf = cp.asnumpy(aotf)
    #     m = max(aotf.flatten())
    #     aotf_z = []
    #     if self.acc == 3:
    #         aotf = cp.array(aotf)
    #     for x in range(self.Nzn):
    #         aotf_z.append(self.xp.sum(aotf[x]))
    #     self.aotf_x = self.xp.log(
    #         aotf[:, int(self.Nn / 2), :].squeeze() + 0.0001)  # cross section perpendicular to x axis
    #     if self.acc == 3:
    #         self.aotf_x = cp.asnumpy(self.aotf_x)
    #     # aotf_x is the same as aotf_y
    #     # self.aotf_y = self.xp.log(aotf[:, :, int(self.Nn / 2)].squeeze() + 0.0001)
    #     yield "3d otf calculated"
    #
    #     if (self.acc == 0) | (self.acc == 3):
    #         img = self.xp.zeros((self.Nz * self._nsteps, self.N, self.N), dtype=self.xp.single)
    #     else:
    #         img = torch.empty((self.Nz * self._nsteps, self.N, self.N), dtype=torch.float, device=self._tdev)
    #
    #     for i in range(self._nsteps):
    #         if (self.acc == 0) | (self.acc == 3):
    #             ootf = self.xp.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
    #             img[self.xp.arange(i, self.Nz * self._nsteps, self._nsteps), :, :] = self.xp.abs(
    #                 self.xp.fft.ifftn(ootf, (self.Nz, self.N, self.N)))
    #         else:
    #             ootf = torch.fft.fftshift(torch.as_tensor(otf, device=self._tdev), ) * self.phasetilts[i, :, :, :]
    #             img[torch.arange(i, self.Nz * self._nsteps, self._nsteps), :, :] = (torch.abs(
    #                 torch.fft.ifftn(ootf, (self.Nz, self.N, self.N)))).to(torch.float)
    #     # OK to use abs here as signal should be all positive.
    #     # Abs is required as the result will be complex as the fourier plane cannot be shifted back to zero when oversampling.
    #     # But should reduction in sampling be allowed here(Nz < Nzn)?
    #
    #     stackfilename = f"Raw_img_stack_{self.N}_{self.pol}.tif"
    #     if self.acc == 0:
    #         # raw image stack
    #         # raw image sum along z axis
    #         self.img_sum_z = np.sum(img, axis=0)
    #         # raw image sum along x (or y) axis
    #         self.img_sum_x = np.sum(img, axis=1)
    #         self.img = img
    #     elif self.acc == 1:
    #         self.img_sum_z = (torch.sum(img, axis=0)).numpy()
    #         self.img_sum_x = (torch.sum(img, axis=1)).numpy()
    #         self.img = img.numpy()
    #     elif self.acc == 2:
    #         self.img_sum_z = (torch.sum(img, axis=0)).detach().cpu().numpy()
    #         self.img_sum_x = (torch.sum(img, axis=1)).detach().cpu().numpy()
    #         self.img = img.detach().cpu().numpy()
    #     elif self.acc == 3:
    #         self.img_sum_z = cp.asnumpy(cp.sum(img, axis=0))
    #         self.img_sum_x = cp.asnumpy(cp.sum(img, axis=1))
    #         self.img = cp.asnumpy(img)
    #
    #     # Save generated images
    #     tifffile.imwrite(stackfilename, self.img)
    #     yield "file saved"
    #
    #     yield f'Finished, Phase tilts calculation time:  {self.elapsed_time:3f}s'

    def raw_image_stack_brownian(self):
        # Calculates 3d psf and otf before the image stack
        self.initialise()

        # Calculating psf
        if self.psf_calc == 'vector':
            psf = self.get_vector_psf()
        else:
            psf = self.get_scalar_psf()
        yield "psf calculated"
        self.psffile = np.sum(psf, axis=(1, 2))

        # Calculating 3d otf
        psf = self.xp.fft.fftshift(psf, axes=0)  # need to set plane zero as in-focus here
        self.psf_z0 = psf[int(self.Nzn / 2 + 5), :, :]  # psf at z=0
        if self.acc == 3:
            self.psf_z0 = cp.asnumpy(self.psf_z0)
        otf = self.xp.fft.fftn(psf)
        aotf = abs(self.xp.fft.fftshift(otf))  # absolute otf
        if self.acc == 3:
            aotf = cp.asnumpy(aotf)
        m = max(aotf.flatten())
        aotf_z = []
        for x in range(self.Nzn):
            aotf_z.append(np.sum(aotf[x]))
        self.aotf_x = np.log(
            aotf[:, int(self.Nn / 2), :].squeeze() + 0.0001)  # cross section perpendicular to x axis
        if self.acc == 3:
            self.aotf_x = cp.asnumpy(self.aotf_x)
        # aotf_x is the same as aotf_y
        # self.aotf_y = self.xp.log(aotf[:, :, int(self.Nn / 2)].squeeze() + 0.0001)
        yield "3d otf calculated"

        self._nsteps = self._phaseStep * self._angleStep
        self.points[:, 0] -= self.xdrift * self.tpoints / 2000
        self.points[:, 2] -= self.zdrift * self.tpoints / 2000
        if (self.acc == 0) | (self.acc == 3):
            img = self.xp.zeros((int(self.tpoints), self.N, self.N), dtype=np.single)
        else:
            img = torch.empty((int(self.tpoints), self.N, self.N), dtype=torch.float, device=self._tdev)

        start_Brownian = time.time()
        tplane = 0
        yield "Starting stack"
        for t in np.arange(0, self.tpoints, self._nsteps):
            for msg in self.phase_tilts():
                yield f'planes at time point = {t} of {self.tpoints}, {msg}'
            # self.points[:, 2] += self.dzn
            for i in range(self._nsteps):
                if (self.acc == 0) | (self.acc == 3):
                    ootf = self.xp.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                    img[tplane, :, :] = self.xp.abs(
                        self.xp.fft.ifft2(self.xp.sum(ootf, axis=0), (self.N, self.N)))
                else:
                    ootf = torch.fft.fftshift(torch.as_tensor(otf, device=self._tdev)) * self.phasetilts[i, :, :, :]
                    img[tplane, :, :] = (torch.abs(
                        torch.fft.ifft2(torch.sum(ootf, axis=0), (self.N, self.N)))).to(torch.float)
                tplane += 1

        # OK to use abs here as signal should be all positive.
        # Abs is required as the result will be complex as the fourier plane cannot be shifted back to zero when oversampling.
        # But should reduction in sampling be allowed here(Nz < Nzn)?

        stackfilename = f"Raw_img_stack_{self.N}_{self.pol}.tif"
        if self.acc == 0:
            # raw image stack
            # raw image sum along z axis
            self.img_sum_z = np.sum(img, axis=0)
            # raw image sum along x (or y) axis
            self.img_sum_x = np.sum(img, axis=1)
            self.img = img
        elif self.acc == 1:
            self.img_sum_z = (torch.sum(img, axis=0)).numpy()
            self.img_sum_x = (torch.sum(img, axis=1)).numpy()
            self.img = img.numpy()
        elif self.acc == 2:
            self.img_sum_z = (torch.sum(img, axis=0)).detach().cpu().numpy()
            self.img_sum_x = (torch.sum(img, axis=1)).detach().cpu().numpy()
            self.img = img.detach().cpu().numpy()
        elif self.acc == 3:
            self.img_sum_z = cp.asnumpy(cp.sum(img, axis=0))
            self.img_sum_x = cp.asnumpy(cp.sum(img, axis=1))
            self.img = cp.asnumpy(img)

        # Save generated images
        tifffile.imwrite(stackfilename, self.img)
        elapsed_Brownian = time.time() - start_Brownian
        yield f'Finished, Phase tilts calculation time:  {elapsed_Brownian:3f}s'
