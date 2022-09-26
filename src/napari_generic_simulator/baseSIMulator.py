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
    pol = None  # polarisation
    acc = None  # acceleration
    psf_calc = None
    _tdev = None
    N = 128  # Points to use in FFT
    pixel_size = 5.5  # Camera pixel size
    magnification = 60  # Objective magnification
    NA = 1.1  # Numerical aperture at sample
    n = 1.33  # Refractive index at sample
    wavelength = 0.52  # Wavelength in um
    npoints = 500  # Number of random points
    zrange = 3.5  # distance either side of focus to calculate, in microns, could be arbitrary
    dz = 0.4  # step size in axial direction of PSF
    fwhmz = 3.0  # FWHM of light sheet in z
    random_seed = 123
    drift = 0.1
    defocus = 1  # de-focus aberration in um
    add_sph = None  # adding primary spherical aberration
    spherical = 0

    def initialise(self):
        np.random.seed(self.random_seed)
        # self.seed(1234)  # set random number generator seed

        self.eta = self.n / self.NA  # right-angle Hex SIM
        self.sigmaz = self.fwhmz / 2.355
        self.dx = self.pixel_size / self.magnification  # Sampling in lateral plane at the sample in um
        self.dxn = self.wavelength / (4 * self.NA)  # 2 * Nyquist frequency in x and y.
        self.Nn = int(np.ceil(self.N * self.dx / self.dxn / 2) * 2)  # Number of points at Nyquist sampling, even number
        self.dxn = self.N * self.dx / self.Nn  # correct spacing
        self.res = self.wavelength / (2 * self.NA)
        oversampling = self.res / self.dxn  # factor by which pupil plane oversamples the coherent psf data
        self.dk = oversampling / (self.Nn / 2)  # Pupil plane sampling
        self.kx, self.ky = np.meshgrid(np.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn),
                                       np.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn))
        self.kr = np.sqrt(self.kx ** 2 + self.ky ** 2)  # Raw pupil function, pupil defined over circle of radius 1.
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
        rad = 5  # radius of sphere of points
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

        if self.acc == 0:
            self.phasetilts = np.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
        elif self.acc == 3:
            self.phasetilts = cp.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
        else:
            self.phasetilts = torch.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=torch.complex64,
                                          device=self._tdev)

        start_time = time.time()

        itcount = 0
        total_its = self._angleStep * self._phaseStep * self.npoints
        lastProg = -1
        self.ph = self.eta * 4 * np.pi * self.NA / self.wavelength
        for astep in range(self._angleStep):
            for pstep in range(self._phaseStep):
                self.points += self.drift * np.random.standard_normal(3)
                for i in range(self.npoints):
                    prog = (100 * itcount) // total_its
                    if prog > lastProg:
                        lastProg = prog
                        yield f'Phase tilts calculation: {prog:.1f}% done'
                    itcount += 1
                    isteps = pstep + self._angleStep * astep  # index of the steps
                    self.x = self.points[i, 0]
                    self.y = self.points[i, 1]
                    z = self.points[i, 2] + self.dz / self._nsteps * isteps
                    self.ph = self.eta * 4 * np.pi * self.NA / self.wavelength
                    self.p1 = pstep * 2 * np.pi / self._phaseStep
                    self.p2 = -pstep * 4 * np.pi / self._phaseStep
                    self._ill()  # gets illumination from the child class
                    if self.pol == 'axial':
                        ill = self._illAx
                    elif self.pol == 'circular':
                        ill = self._illCi
                    else:
                        ill = self._illIp
                    if self.acc == 0:
                        px = np.exp(1j * np.single(self.x * self.kxy))[:, np.newaxis]
                        py = np.exp(1j * np.single(self.y * self.kxy))
                        pz = (np.exp(1j * np.single(z * self.kz)) * ill)[:, np.newaxis, np.newaxis]
                        self.phasetilts[isteps, :, :, :] += (px * py) * pz
                    elif self.acc == 3:
                        px = cp.array(np.exp(1j * np.single(self.x * self.kxy))[:, np.newaxis])
                        py = cp.array(np.exp(1j * np.single(self.y * self.kxy)))
                        pz = cp.array((np.exp(1j * np.single(z * self.kz)) * ill)[:, np.newaxis, np.newaxis])
                        self.phasetilts[isteps, :, :, :] += (px * py) * pz
                    else:
                        px = torch.as_tensor(np.exp(1j * np.single(self.x * self.kxy)), device=self._tdev)
                        py = torch.as_tensor(np.exp(1j * np.single(self.y * self.kxy)), device=self._tdev)
                        pz = torch.as_tensor((np.exp(1j * np.single(z * self.kz)) * ill),
                                             device=self._tdev)
                        self.phasetilts[isteps, :, :, :] += (px[..., None] * py) * pz[..., None, None]
        self.elapsed_time = time.time() - start_time
        yield f'Phase tilts calculation time:  {self.elapsed_time:3f}s'

    def get_vector_psf(self):
        k0 = 2 * np.pi * self.n / self.wavelength
        krmax = self.NA * k0 / self.n
        kx = krmax * (self.kx + 1e-15)
        ky = krmax * (self.ky + 1e-15)
        kr2 = (kx ** 2 + ky ** 2)  # square kr
        e_in = 1.0 * (kr2 < krmax ** 2)
        kz = np.sqrt((k0 ** 2 - kr2) + 0j)

        # Calculating psf
        nz = 0
        psf = np.zeros((self.Nzn, self.Nn, self.Nn))
        pupil = self.kr < 1

        # calculate intensity of random arrangement of dipoles excited by a given polarisation s
        # p are the vertices of an dodecahedron
        p0 = np.reshape(np.array([0, 1, 0]), (1, 3))
        p1 = np.reshape(np.array([-0.666666, 0., 0.745353,
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
        p2 = np.reshape(np.array([-1.37638, 0., 0.262866,
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
        p2 = p2 / np.linalg.norm(p2[0, :])

        p = p1
        s1 = np.array([1, 0, 0])  # x polarised illumination orientation
        excitation1 = (s1 @ p.T) ** 2
        s2 = np.array([0, 1, 0])  # y polarised illumination orientation
        excitation2 = (s2 @ p.T) ** 2
        s3 = np.array([0, 0, 1])  # z polarised illumination orientation
        excitation3 = (s3 @ p.T) ** 2

        for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
            fx1 = k0 * (k0 * ky ** 2 + kx ** 2 * kz) / (np.sqrt(kz / k0) * kr2) * e_in * np.exp(1j * z * kz)
            Exx = np.fft.fftshift(np.fft.fft2(fx1))  # x-polarised field at camera for x-oriented dipole
            fy1 = k0 * kx * ky * (kz - k0) / (np.sqrt(kz / k0) * kr2) * e_in * np.exp(1j * z * kz)
            Exy = np.fft.fftshift(np.fft.fft2(fy1))  # y-polarised field at camera for x-oriented dipole
            fx2 = k0 * kx * ky * (kz - k0) / (np.sqrt(kz / k0) * kr2) * e_in * np.exp(1j * z * kz)
            Eyx = np.fft.fftshift(np.fft.fft2(fx2))  # x-polarised field at camera for y-oriented dipole
            fy2 = k0 * (k0 * kx ** 2 + ky ** 2 * kz) / (np.sqrt(kz / k0) * kr2) * e_in * np.exp(1j * z * kz)
            Eyy = np.fft.fftshift(np.fft.fft2(fy2))  # y-polarised field at camera for y-oriented dipole
            fx3 = k0 * kx / np.sqrt(kz / k0) * e_in * np.exp(1j * z * kz)
            Ezx = np.fft.fftshift(np.fft.fft2(fx3))  # x-polarised field at camera for z-oriented dipole
            fy3 = k0 * ky / np.sqrt(kz / k0) * e_in * np.exp(1j * z * kz)
            Ezy = np.fft.fftshift(np.fft.fft2(fy3))  # y-polarised field at camera for z-oriented dipole
            intensityx = np.zeros((self.Nn, self.Nn))
            intensityy = np.zeros((self.Nn, self.Nn))
            intensityz = np.zeros((self.Nn, self.Nn))
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
            psf[nz, :, :] = intensity * np.exp(-z ** 2 / 2 / self.sigmaz ** 2)
            nz = nz + 1
        print(np.sum(psf, axis=(1, 2)))
        sumpsf = np.sum(psf[self.Nzn // 2], axis=(0, 1))
        psf = psf / sumpsf
        print(np.sum(psf, axis=(1, 2)))
        return psf

    def get_scalar_psf(self):
        nz = 0
        psf = np.zeros((self.Nzn, self.Nn, self.Nn))
        pupil = self.kr < 1
        for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
            c = (np.exp(
                1j * ((z + self.defocus) * self.n * 2 * np.pi / self.wavelength *
                      np.sqrt(1 - (self.kr * pupil) ** 2 * self.NA ** 2 / self.n ** 2) + self.spherical))) * pupil
            psf[nz, :, :] = abs(np.fft.fftshift(np.fft.ifft2(c))) ** 2 * np.exp(-z ** 2 / 2 / self.sigmaz ** 2)
            nz = nz + 1
        # Normalised so power in resampled psf(see later on) is unity in focal plane
        print(np.sum(psf, axis = (1,2)))
        psf = psf * self.Nn ** 2 / np.sum(pupil) * self.Nz / self.Nzn
        self.psf_z0 = psf[int(self.Nzn / 2 + 5), :, :]  # psf at z=0
        print(np.sum(psf, axis = (1,2)))
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
        yield "psf calculated"

        # Calculating 3d otf
        otf = np.fft.fftn(psf)
        aotf = abs(np.fft.fftshift(otf))  # absolute otf
        m = max(aotf.flatten())
        aotf_z = []
        for x in range(self.Nzn):
            aotf_z.append(np.sum(aotf[x]))
        self.aotf_x = np.log(
            aotf[:, int(self.Nn / 2), :].squeeze() + 0.0001)  # cross section perpendicular to x axis
        self.aotf_y = np.log(aotf[:, :, int(self.Nn / 2)].squeeze() + 0.0001)
        yield "3d otf calculated"

        if self.acc == 0:
            img = np.zeros((self.Nz * self._nsteps, self.N, self.N), dtype=np.single)
        elif self.acc == 3:
            img = cp.zeros((self.Nz * self._nsteps, self.N, self.N), dtype=np.single)
        else:
            img = torch.empty((self.Nz * self._nsteps, self.N, self.N), dtype=torch.float, device=self._tdev)

        for i in range(self._nsteps):
            if self.acc == 0:
                ootf = np.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                img[np.arange(i, self.Nz * self._nsteps, self._nsteps), :, :] = abs(
                    np.fft.ifftn(ootf, (self.Nz, self.N, self.N)))
            elif self.acc == 3:
                ootf = cp.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                img[cp.arange(i, self.Nz * self._nsteps, self._nsteps), :, :] = cp.abs(
                    cp.fft.ifftn(ootf, (self.Nz, self.N, self.N)))
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
        psf = self.get_scalar_psf()
        yield "psf calculated"

        # Calculating 3d otf
        psf = np.fft.fftshift(psf, axes=0)  # need to set plane zero as in-focus here
        otf = np.fft.fftn(psf)
        aotf = abs(np.fft.fftshift(otf))  # absolute otf
        m = max(aotf.flatten())
        aotf_z = []
        for x in range(self.Nzn):
            aotf_z.append(np.sum(aotf[x]))
        self.aotf_x = np.log(
            aotf[:, int(self.Nn / 2), :].squeeze() + 0.0001)  # cross section perpendicular to x axis
        self.aotf_y = np.log(aotf[:, :, int(self.Nn / 2)].squeeze() + 0.0001)
        yield "3d otf calculated"

        self._nsteps = self._phaseStep * self._angleStep
        self.points[:, 2] -= self.zrange
        if self.acc == 0:
            img = np.zeros((self.Nz * self._nsteps, self.N, self.N), dtype=np.single)
        elif self.acc == 3:
            img = cp.zeros((self.Nz * self._nsteps, self.N, self.N), dtype=np.single)
        else:
            img = torch.empty((self.Nz * self._nsteps, self.N, self.N), dtype=torch.float, device=self._tdev)

        start_Brownian = time.time()
        zplane = 0
        for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
            for msg in self.phase_tilts():
                yield f'planes at z={z:.2f}, {msg}'
            self.points[:, 2] += self.dzn
            for i in range(self._nsteps):
                if self.acc == 0:
                    ootf = np.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                    img[zplane, :, :] = abs(
                        np.fft.ifft2(np.sum(ootf, axis=0), (self.N, self.N)))
                elif self.acc == 3:
                    ootf = cp.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                    img[zplane, :, :] = cp.abs(
                        cp.fft.ifft2(cp.sum(ootf, axis=0), (self.N, self.N)))
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
