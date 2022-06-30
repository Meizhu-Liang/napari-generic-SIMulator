"""
The parent class to simulate raw data for SIM. @author: Meizhu Liang @Imperial College
Some calculations are adapted by work by Mark Neil @Imperial College
"""

from random import seed
import numpy as np
import time
import tifffile

try:
    import cupy as cp
    print('cupy imported')
    import_cp = True
except Exception as e:
    print(str(e))
    import_cp = False

class Base_simulator:

    axial = True
    use_cupy = True
    N = 512  # Points to use in FFT
    pixel_size = 5.5  # Camera pixel size
    magnification = 60  # Objective magnification
    dx = pixel_size / magnification  # Sampling in lateral plane at the sample in um
    NA = 1.1  # Numerical aperture at sample
    n = 1.33  # Refractive index at sample
    wavelength = 0.52  # Wavelength in um
    npoints = 500  # Number of random points
    seed(1234)  # set random number generator seed
    eta = n / NA  # right-angle Hex SIM
    zrange = 7.0  # distance either side of focus to calculate, in microns, could be arbitrary
    dz = 0.4  # step size in axial direction of PSF
    fwhmz = 3.0  # FWHM of light sheet in z
    sigmaz = fwhmz / 2.355
    dxn = wavelength / (4 * NA)  # 2 * Nyquist frequency in x and y.
    Nn = int(np.ceil(N * dx / dxn / 2) * 2)  # Number of points at Nyquist sampling, even number
    dxn = N * dx / Nn  # correct spacing
    res = wavelength / (2 * NA)
    oversampling = res / dxn  # factor by which pupil plane oversamples the coherent psf data
    dk = oversampling / (Nn / 2)  # Pupil plane sampling
    kx, ky = np.meshgrid(np.linspace(-dk * Nn / 2, dk * Nn / 2 - dk, Nn),
                         np.linspace(-dk * Nn / 2, dk * Nn / 2 - dk, Nn))
    kr = np.sqrt(kx ** 2 + ky ** 2)  # Raw pupil function, pupil defined over circle of radius 1.
    csum = sum(sum((kr < 1)))  # normalise by csum so peak intensity is 1

    alpha = np.arcsin(NA / n)
    # Nyquist sampling in z, reduce by 10 % to account for gaussian light sheet
    dzn = 0.8 * wavelength / (2 * n * (1 - np.cos(alpha)))
    Nz = 2 * np.ceil(zrange / dz)
    dz = 2 * zrange / Nz
    Nzn = int(2 * np.ceil(zrange / dzn))
    dzn = 2 * zrange / Nzn
    if Nz < Nzn:
        Nz = Nzn
        dz = dzn

    def point_cloud(self):
        print("Calculating point cloud")

        rad = 10  # radius of sphere of points
        # Multiply the points several timesto get the enough number
        pointsxn = (2 * np.random.rand(self.npoints * 3, 3) - 1) * [rad, rad, rad]

        pointsxnr = np.sum(pointsxn * pointsxn, axis=1)  # multiple times the points
        points_sphere = pointsxn[pointsxnr < (rad ** 2), :]  # simulate spheres from cubes
        self.points = points_sphere[(range(self.npoints)), :]
        self.points[:, 2] = self.points[:, 2] / 2  # to make the point cloud for OTF a ellipsoid rather than a sphere

    def phase_tilts(self):
        """Generate phase tilts in frequency space"""

        xyrange = self.Nn / 2 * self.dxn
        dkxy = np.pi / xyrange
        self.kxy = np.arange(-self.Nn / 2 * dkxy, (self.Nn / 2) * dkxy, dkxy)
        dkz = np.pi / self.zrange
        self.kz = np.arange(-self.Nzn / 2 * dkz, (self.Nzn / 2) * dkz, dkz)

        if self.use_cupy:
            self.phasetilts = cp.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
        else:
            self.phasetilts = np.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=np.complex64)

        print("Calculating pointwise phase tilts")

        start_time = time.time()

        for self.xx in range(self._nsteps):
            if self.use_cupy:
                pxyz = cp.zeros((self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
            else:
                pxyz = np.zeros((self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
            for i in range(self.npoints):
                self.x = self.points[i, 0]
                self.y = self.points[i, 1]
                z = self.points[i, 2] + self.dz / self._nsteps * (self.xx - 1)
                self.ph = self.eta * 4 * np.pi * self.NA / self.wavelength
                self.p1 = -self.xx * 2 * np.pi / self._nsteps
                self.p2 = self.xx * 4 * np.pi / self._nsteps
                self._ill()
                if self.axial:  # axial polarisation normalised to peak intensity of 1
                    ill = self._illAx
                else:  # in plane polarisation normalised to peak intensity of 1
                    ill = self._illIp
                if self.use_cupy:
                    px = cp.array(np.exp(1j * np.single(self.x * self.kxy)))
                    py = cp.array(np.exp(1j * np.single(self.y * self.kxy)))
                    pz = cp.array(np.exp(1j * np.single(z * self.kz)) * ill)
                    pxy = cp.array(px[:, np.newaxis] * py)
                    for ii in range(len(self.kz)):
                        pxyz[ii, :, :] = pxy * pz[ii]
                    self.phasetilts[self.xx, :, :, :] = self.phasetilts[self.xx, :, :, :] + pxyz
                else:
                    px = np.exp(1j * np.single(self.x * self.kxy))
                    py = np.exp(1j * np.single(self.y * self.kxy))
                    pz = np.exp(1j * np.single(z * self.kz)) * ill
                    pxy = px[:, np.newaxis] * py
                    for ii in range(len(self.kz)):
                        pxyz[ii, :, :] = pxy * pz[ii]
                    self.phasetilts[self.xx, :, :, :] = self.phasetilts[self.xx, :, :, :] + pxyz

        elapsed_time = time.time() - start_time
        print(f'Phase tilts calculation:  {elapsed_time:.3f}s')

    def raw_image_stack(self):
        # Calculates point cloud, phase tilts, 3d psf and otf before the image stack

        self.point_cloud()
        self.phase_tilts()
        print("Calculating 3d psf")
        nz = 0
        psf = np.zeros((self.Nzn, self.Nn, self.Nn))
        pupil = self.kr < 1

        for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
            c = (np.exp(
                1j * (z * self.n * 2 * np.pi / self.wavelength *
                      np.sqrt((1 - (self.kr * pupil) ** 2 * self.NA ** 2 / self.n ** 2))))) * pupil
            psf[nz, :, :] = abs(np.fft.fftshift(np.fft.ifft2(c))) ** 2 * np.exp(-z ** 2 / 2 / self.sigmaz ** 2)
            nz = nz + 1

        # Normalised so power in resampled psf(see later on) is unity in focal plane
        psf = psf * self.Nn ** 2 / np.sum(pupil) * self.Nz / self.Nzn

        print("Calculating 3d otf")
        psf = psf
        otf = np.fft.fftn(psf)
        aotf = abs(np.fft.fftshift(otf))
        m = max(aotf.flatten())
        aotf_z = []
        for x in range(self.Nzn):
            aotf_z.append(np.sum(aotf[x]))

        print("Calculating raw image stack")
        if self.use_cupy:
            img = cp.zeros((self.Nz * self._nsteps, self.N, self.N), dtype=np.single)
        else:
            img = np.zeros((self.Nz * self._nsteps, self.N, self.N), dtype=np.single)
        for i in range(self._nsteps):
            if self.use_cupy:
                ootf = cp.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                img[cp.arange(i, self.Nz * self._nsteps, self._nsteps), :, :] = cp.abs(cp.fft.ifftn(ootf, (self.Nz, self.N, self.N)))
            else:
                ootf = np.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                img[np.arange(i, self.Nz * self._nsteps, self._nsteps), :, :] = np.abs(np.fft.ifftn(ootf, (self.Nz, self.N, self.N)))
        # OK to use abs here as signal should be all positive.
        # Abs is required as the result will be complex as the fourier plane cannot be shifted back to zero when oversampling.
        # But should reduction in sampling be allowed here(Nz < Nzn)?

        # raw image stack
        if self.use_cupy:
            img = cp.asnumpy(img)

        # raw image sum along z axis
        if self.use_cupy:
            img_sum_z = cp.asnumpy(cp.sum(img, axis=0))
        else:
            img_sum_z = np.sum(img, axis=0)

        # raw image sum along x (or y) axis"
        if self.use_cupy:
            img_sum_x = cp.asnumpy(cp.sum(img, axis=1))
        else:
            img_sum_x = np.sum(img, axis=1)

        # Save generated images
        if self.axial:
            stackfilename = "Raw_img_stack90_512_axial.tif"
        else:
            stackfilename = "Raw_img_stack90_512_inplane.tif"
        if self.use_cupy:
            tifffile.imwrite(stackfilename, cp.asnumpy(img))
        else:
            tifffile.imwrite(stackfilename, img)
        print('Raw image stack saved')

        # Return img_sum_z, img_sum_x, psf_xy, otf_yz, otf_xz, otf_sum_xy (otf sum in x-y plane)
        return img, img_sum_z, img_sum_x, psf[int(self.Nzn / 2 + 10), :, :], np.log(aotf[:, int(self.Nn / 2), :].squeeze() + 0.0001), \
               np.log(aotf[:, :, int(self.Nn / 2)].squeeze()) + 0.0001


