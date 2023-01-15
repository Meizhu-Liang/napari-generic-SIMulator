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
    xp = np
    pol = None  # polarisation
    acc = None  # acceleration
    psf_calc = None
    _tdev = None
    N = 512  # Points to use in FFT
    pixel_size = 5.5  # Camera pixel size
    magnification = 60  # Objective magnification
    NA = 1.1  # Numerical aperture at sample
    n = 1.33  # Refractive index at sample
    wavelength = 0.52  # Wavelength in um
    npoints = 500  # Number of random points
    zrange = 7.0  # distance either side of focus to calculate, in microns, could be arbitrary
    dz = 0.4  # step size in axial direction of PSF
    fwhmz = 3.0  # FWHM of light sheet in z
    random_seed = 123
    drift = 0
    defocus = 0  # de-focus aberration in um
    # add_sph = None  # adding primary spherical aberration
    spherical = 0
    sph_abb = 0

    def initialise(self):
        if self.acc == 3:
            self.xp = cp
        np.random.seed(self.random_seed)
        # self.seed(1234)  # set random number generator seed
        self.sigmaz = self.fwhmz / 2.355
        self.dx = self.pixel_size / self.magnification  # Sampling in lateral plane at the sample in um
        self.dxn = self.wavelength / (4 * self.NA)  # 2 * Nyquist frequency in x and y.
        self.Nn = int(np.ceil(self.N * self.dx / self.dxn / 2) * 2)  # Number of points at Nyquist sampling, even number
        self.dxn = self.N * self.dx / self.Nn  # correct spacing
        self.res = self.wavelength / (2 * self.NA)
        oversampling = self.res / self.dxn  # factor by which pupil plane oversamples the coherent psf data
        self.dk = oversampling / (self.Nn / 2)  # Pupil plane sampling
        self.k0 = 2 * np.pi * self.n / self.wavelength
        self.kx, self.ky = self.xp.meshgrid(self.xp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn),
                                       self.xp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn))
        self.kr = np.sqrt(self.kx ** 2 + self.ky ** 2)  # Raw pupil function, pupil defined over circle of radius 1.
        self.krmax = self.NA * self.k0 / self.n
        self.kr2 = self.kx ** 2 + self.ky ** 2
        self.spherical = self.sph_abb * np.sqrt(5) * (6 * (self.kr ** 4 - self.kr ** 2) + 1)
        self.csum = sum(sum((self.kr < 1)))  # normalise by csum so peak intensity is 1

        self.alpha = np.arcsin(self.NA / self.n)
        # Nyquist sampling in z, reduce by 10 % to account for gaussian light sheet
        self.dzn = 0.8 * self.wavelength / (2 * self.n * (1 - np.cos(self.alpha)))
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
        self._tdev = torch.device('cuda' if self.acc == 2 else 'cpu')

    def point_cloud(self):
        """
        Generates a point-cloud as the object in the imaging system.
        """
        rad = 10  # radius of sphere of points
        # Multiply the points several times to get the enough number
        pointsxn = (2 * np.random.rand(self.npoints * 3, 3) - 1) * [rad, rad, rad]

        pointsxnr = np.sum(pointsxn * pointsxn, axis=1)  # multiple times the points
        points_sphere = pointsxn[pointsxnr < (rad ** 2), :]  # simulate spheres from cubes
        self.points = points_sphere[(range(self.npoints)), :]
        self.points[:, 2] = self.points[:, 2] / 2  # to make the point cloud for OTF a ellipsoid rather than a sphere

    def phase_tilts(self):
        """Generates phase tilts in frequency space"""
        xyrange = self.Nn / 2 * self.dxn
        dkxy = np.pi / xyrange
        dkz = np.pi / self.zrange
        self.kxy = np.arange(-self.Nn / 2 * dkxy, (self.Nn / 2) * dkxy, dkxy)
        self.kz = np.arange(-self.Nzn / 2 * dkz, (self.Nzn / 2) * dkz, dkz)

        if (self.acc == 0) | (self.acc == 3):
            self.phasetilts = self.xp.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=self.xp.complex64)
        else:
            self.phasetilts = torch.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=torch.complex64,
                                          device=self._tdev)

        start_time = time.time()

        itcount = 0
        total_its = self._angleStep * self._phaseStep * self.npoints
        lastProg = -1
        self.ph = self._eta * 4 * np.pi * self.NA / self.wavelength
        for pstep in range(self._phaseStep):
            for astep in range(self._angleStep):
                self.points += self.drift * np.random.standard_normal(3)
                isteps = pstep + self._angleStep * astep  # index of the steps
                for i in range(self.npoints):
                    prog = (100 * itcount) // total_its
                    if prog > lastProg:
                        lastProg = prog
                        yield f'Phase tilts calculation: {prog:.1f}% done'
                    itcount += 1
                    self.x = self.points[i, 0]
                    self.y = self.points[i, 1]
                    z = self.points[i, 2] + self.dz / self._nsteps * isteps
                    self.ph = self._eta * 4 * np.pi * self.NA / self.wavelength
                    self.p1 = pstep * 2 * np.pi / self._phaseStep
                    self.p2 = -pstep * 4 * np.pi / self._phaseStep
                    if self.pol == 'axial':
                        # get illumination from the child class
                        ill = self._illAx()
                    elif self.pol == 'circular':
                        ill = self._illCi()
                    else:
                        ill = self._illIp()
                    if self.acc == 0:
                        px = np.exp(1j * np.single(self.x * self.kxy))[:, np.newaxis]
                        py = np.exp(1j * np.single(self.y * self.kxy))
                        pz = (np.exp(1j * np.single(z * self.kz)) * ill[astep])[:, np.newaxis, np.newaxis]
                        self.phasetilts[isteps, :, :, :] += (px * py) * pz
                    elif self.acc == 3:
                        px = cp.array(np.exp(1j * np.single(self.x * self.kxy))[:, np.newaxis])
                        py = cp.array(np.exp(1j * np.single(self.y * self.kxy)))
                        pz = cp.array((np.exp(1j * np.single(z * self.kz)) * ill[astep])[:, np.newaxis, np.newaxis])
                        self.phasetilts[isteps, :, :, :] += (px * py) * pz
                    else:
                        px = torch.as_tensor(np.exp(1j * np.single(self.x * self.kxy)), device=self._tdev)
                        py = torch.as_tensor(np.exp(1j * np.single(self.y * self.kxy)), device=self._tdev)
                        pz = torch.as_tensor((np.exp(1j * np.single(z * self.kz)) * ill[astep]),
                                             device=self._tdev)
                        self.phasetilts[isteps, :, :, :] += (px[..., None] * py) * pz[..., None, None]
        self.elapsed_time = time.time() - start_time
        yield f'Phase tilts calculation time:  {self.elapsed_time:3f}s'

    def get_vector_psf(self):
        # use krmax to define the pupil function
        kx = self.krmax * (self.kx + 1e-15)
        ky = self.krmax * (self.ky + 1e-15)
        kr2 = (kx ** 2 + ky ** 2)  # square kr
        e_in = 1.0 * (kr2 < self.krmax ** 2)
        kz = np.sqrt((self.k0 ** 2 - kr2) + 0j)

        # Calculating psf
        nz = 0
        psf = self.xp.zeros((self.Nzn, self.Nn, self.Nn))
        pupil = self.kr < 1

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

        p = p1
        s1 = self.xp.array([1, 0, 0])  # x polarised illumination orientation
        excitation1 = (s1 @ p.T) ** 2
        s2 = self.xp.array([0, 1, 0])  # y polarised illumination orientation
        excitation2 = (s2 @ p.T) ** 2
        s3 = self.xp.array([0, 0, 1])  # z polarised illumination orientation
        excitation3 = (s3 @ p.T) ** 2

        for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
            fx1 = self.k0 * (self.k0 * ky ** 2 + kx ** 2 * kz) / (self.xp.sqrt(kz / self.k0) * kr2) * e_in * np.exp(1j * z * kz)
            Exx = self.xp.fft.fftshift(self.xp.fft.fft2(fx1))  # x-polarised field at camera for x-oriented dipole
            fy1 = self.k0 * kx * ky * (kz - self.k0) / (self.xp.sqrt(kz / self.k0) * kr2) * e_in * np.exp(1j * z * kz)
            Exy = self.xp.fft.fftshift(self.xp.fft.fft2(fy1))  # y-polarised field at camera for x-oriented dipole
            fx2 = self.k0 * kx * ky * (kz - self.k0) / (self.xp.sqrt(kz / self.k0) * kr2) * e_in * np.exp(1j * z * kz)
            Eyx = self.xp.fft.fftshift(self.xp.fft.fft2(fx2))  # x-polarised field at camera for y-oriented dipole
            fy2 = self.k0 * (self.k0 * kx ** 2 + ky ** 2 * kz) / (self.xp.sqrt(kz / self.k0) * kr2) * e_in * np.exp(1j * z * kz)
            Eyy = self.xp.fft.fftshift(self.xp.fft.fft2(fy2))  # y-polarised field at camera for y-oriented dipole
            fx3 = self.k0 * kx / self.xp.sqrt(kz / self.k0) * e_in * np.exp(1j * z * kz)
            Ezx = self.xp.fft.fftshift(self.xp.fft.fft2(fx3))  # x-polarised field at camera for z-oriented dipole
            fy3 = self.k0 * ky / self.xp.sqrt(kz / self.k0) * e_in * np.exp(1j * z * kz)
            Ezy = self.xp.fft.fftshift(self.xp.fft.fft2(fy3))  # y-polarised field at camera for z-oriented dipole
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
        return psf

    def get_scalar_psf(self):
        # use krmax to define the pupil function
        kx = self.krmax * self.kx
        ky = self.krmax * self.ky
        kr2 = (kx ** 2 + ky ** 2)  # square kr
        2 * np.pi * self.n / self.wavelength
        kz = self.xp.sqrt((self.k0 ** 2 - kr2) + 0j)
        nz = 0
        psf = self.xp.zeros((self.Nzn, self.Nn, self.Nn))
        pupil = self.kr < 1
        for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
            # c = (self.xp.exp(1j * ((z + self.defocus) * kz + self.spherical))) * pupil
            # psf[nz, :, :] = abs(self.xp.fft.fftshift(self.xp.fft.ifft2(c))) ** 2 * self.xp.exp(-z ** 2 / 2 / self.sigmaz ** 2)
            c = (np.exp(
                1j * (z * self.n * 2 * np.pi / self.wavelength *
                      np.sqrt(1 - (self.kr * pupil) ** 2 * self.NA ** 2 / self.n ** 2)))) * pupil
            psf[nz, :, :] = abs(np.fft.fftshift(np.fft.ifft2(c))) ** 2 * np.exp(-z ** 2 / 2 / self.sigmaz ** 2)
            nz = nz + 1
        # Normalised so power in resampled psf(see later on) is unity in focal plane
        psf = psf * self.Nn ** 2 / self.xp.sum(pupil) * self.Nz / self.Nzn
        return psf

    def raw_image_stack(self):
        # Calculates point cloud, phase tilts, 3d psf and otf before the image stack.
        self.initialise()
        self.drift = 0.0  # no random walk using this method
        self.point_cloud()
        yield "Point cloud calculated"

        self._nsteps = self._phaseStep * self._angleStep
        for msg in self.phase_tilts():
            yield msg
        if self.psf_calc == 'vector':
            psf = self.get_vector_psf()
        else:
            psf = self.get_scalar_psf()
        self.psf_z0 = psf[int(self.Nzn / 2 + 5), :, :]  # psf at z=0
        if self.acc == 3:
            self.psf_z0 = cp.asnumpy(self.psf_z0)
        yield "psf calculated"

        # Calculating 3d otf
        otf = self.xp.fft.fftn(psf)
        aotf = abs(self.xp.fft.fftshift(otf))  # absolute otf
        if self.acc == 3:
            aotf = cp.asnumpy(aotf)
        m = max(aotf.flatten())
        aotf_z = []
        if self.acc == 3:
            aotf = cp.array(aotf)
        for x in range(self.Nzn):
            aotf_z.append(self.xp.sum(aotf[x]))
        self.aotf_x = self.xp.log(
            aotf[:, int(self.Nn / 2), :].squeeze() + 0.0001)  # cross section perpendicular to x axis
        if self.acc == 3:
            self.aotf_x = cp.asnumpy(self.aotf_x)
        # aotf_x is the same as aotf_y
        # self.aotf_y = self.xp.log(aotf[:, :, int(self.Nn / 2)].squeeze() + 0.0001)
        yield "3d otf calculated"

        if (self.acc == 0) | (self.acc == 3):
            img = self.xp.zeros((self.Nz * self._nsteps, self.N, self.N), dtype=self.xp.single)
        else:
            img = torch.empty((self.Nz * self._nsteps, self.N, self.N), dtype=torch.float, device=self._tdev)

        for i in range(self._nsteps):
            if (self.acc == 0) | (self.acc == 3):
                ootf = self.xp.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                img[self.xp.arange(i, self.Nz * self._nsteps, self._nsteps), :, :] = self.xp.abs(
                    self.xp.fft.ifftn(ootf, (self.Nz, self.N, self.N)))
            else:
                ootf = torch.fft.fftshift(torch.as_tensor(otf, device=self._tdev), ) * self.phasetilts[i, :, :, :]
                img[torch.arange(i, self.Nz * self._nsteps, self._nsteps), :, :] = (torch.abs(
                    torch.fft.ifftn(ootf, (self.Nz, self.N, self.N)))).to(torch.float)
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
        yield "file saved"

        yield f'Finished, Phase tilts calculation time:  {self.elapsed_time:3f}s'

    def raw_image_stack_brownian(self):
        # Calculates point cloud, phase tilts, 3d psf and otf before the image stack
        self.initialise()
        self.point_cloud()
        yield "Point cloud calculated"

        # Calculating psf
        if self.psf_calc == 'vector':
            psf = self.get_vector_psf()
        else:
            psf = self.get_scalar_psf()
        yield "psf calculated"

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
        self.points[:, 2] -= self.zrange
        if (self.acc == 0) | (self.acc == 3):
            img = self.xp.zeros((self.Nz * self._nsteps, self.N, self.N), dtype=np.single)
        else:
            img = torch.empty((self.Nz * self._nsteps, self.N, self.N), dtype=torch.float, device=self._tdev)

        start_Brownian = time.time()
        zplane = 0
        for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
            for msg in self.phase_tilts():
                yield f'planes at z={z:.2f}, {msg}'
            self.points[:, 2] += self.dzn
            for i in range(self._nsteps):
                if (self.acc == 0) | (self.acc == 3):
                    ootf = self.xp.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                    img[zplane, :, :] = self.xp.abs(
                        self.xp.fft.ifft2(self.xp.sum(ootf, axis=0), (self.N, self.N)))
                else:
                    ootf = torch.fft.fftshift(torch.as_tensor(otf, device=self._tdev)) * self.phasetilts[i, :, :, :]
                    img[zplane, :, :] = (torch.abs(
                        torch.fft.ifft2(torch.sum(ootf, axis=0), (self.N, self.N)))).to(torch.float)
                zplane += 1

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
