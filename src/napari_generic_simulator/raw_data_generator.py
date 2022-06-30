"""
@author: Meizhu Liang @Imperial College, Mark Neil @Imperial College
"""

import numpy as np
from matplotlib import pyplot as plt
from random import seed
import time
import tifffile

try:
    import cupy as cp
    import_cp = True
    print('cupy imported')
except ImportError:
    import_cp = False


# try:
#     import torch
#     pytorch = True
#     print('pytorch imported')
# except ImportError:
#     pytorch = False




class HexSIM_simulator():
    """
    Class to simulate raw data for Hexagonal SIM at regular and right angles.
    """
    def __init__(self, mode, axial, use_cupy, N, pixel_size, magnification, NA, n, wavelength, npoints, zrange,
                 dz, fwhmz):
        """
                N: Points to use in FFT
                """
        self.axial = axial
        self.use_cupy = use_cupy
        self.N = N  # Points to use in FFT
        self.pixel_size = pixel_size  # Camera pixel size
        self.magnification = magnification  # Objective magnification
        dx = self.pixel_size / self.magnification  # Sampling in lateral plane at the sample in um
        self.NA = NA  # Numerical aperture at sample
        self.n = n  # Refractive index at sample
        self.wavelength = wavelength  # Wavelength in um
        self.npoints = npoints  # Number of random points
        seed(1234)  # set random number generator seed
        # eta is the factor by which the illumination grid frequency exceeds the incoherent cutoff, eta = 1 for normal SIM,
        # eta=sqrt(3) / 2 to maximise resolution without zeros in TF
        # For a normal SIM, maximum resolution extension = 1 + eta
        # carrier is 2 * kmax * eta
        self.eta = self.n / self.NA  # right-angle Hex SIM
        # self.eta = n * np.sqrt(3.0) / 2 / NA  # Hex SIM
        self.zrange = zrange  # distance either side of focus to calculate, in microns, could be arbitrary
        self.dz = dz  # step size in axial direction of PSF
        self.fwhmz = fwhmz  # FWHM of light sheet in z
        self.sigmaz = self.fwhmz / 2.355
        self.dxn = self.wavelength / (4 * self.NA)  # 2 * Nyquist frequency in x and y.
        self.Nn = int(np.ceil(self.N * dx / self.dxn / 2) * 2)  # Number of points at Nyquist sampling, even number
        self.dxn = self.N * dx / self.Nn  # correct spacing
        res = self.wavelength / (2 * self.NA)
        oversampling = res / self.dxn  # factor by which pupil plane oversamples the coherent psf data
        dk = oversampling / (self.Nn / 2)  # Pupil plane sampling
        kx, ky = np.meshgrid(np.linspace(-dk * self.Nn / 2, dk * self.Nn / 2 - dk, self.Nn),
                             np.linspace(-dk * self.Nn / 2, dk * self.Nn / 2 - dk, self.Nn))
        self.kr = np.sqrt(kx ** 2 + ky ** 2)  # Raw pupil function, pupil defined over circle of radius 1.
        csum = sum(sum((self.kr < 1)))  # normalise by csum so peak intensity is 1

        alpha = np.arcsin(self.NA / self.n)
        # Nyquist sampling in z, reduce by 10 % to account for gaussian light sheet
        self.dzn = 0.8 * self.wavelength / (2 * n * (1 - np.cos(alpha)))
        Nz = 2 * np.ceil(self.zrange / self.dz)
        self.dz = 2 * self.zrange / Nz
        self.Nzn = int(2 * np.ceil(self.zrange / self.dzn))
        self.dzn = 2 * self.zrange / self.Nzn
        if Nz < self.Nzn:
            self.Nz = self.Nzn
            self.dz = self.dzn
        self.mode = mode
        self.n_PhaseStep = 7  # number of phase steps

    @property
    def use_cupy(self):
        return (self._use_cupy)

    @use_cupy.setter
    def use_cupy(self, bool):
        if bool:
            if import_cp:
                pass
            else:
                raise ModuleNotFoundError
        self._use_cupy = bool


    def point_cloud(self):
        print("Calculating point cloud")

        rad = 10  # radius of sphere of points
        # Multiply the points several timesto get the enough number
        pointsxn = (2 * np.random.rand(self.npoints * 3, 3) - 1) * [rad, rad, rad]

        pointsxnr = np.sum(pointsxn * pointsxn, axis=1)
        points_sphere = pointsxn[pointsxnr < (rad ** 2), :]  # simulate spheres from cubes
        self.points = points_sphere[(range(self.npoints)), :]
        self.points[:, 2] = self.points[:, 2] / 2  # to make the point cloud for OTF a ellipsoid rather than a sphere

        # plt.figure(20)
        ax = plt.axes(projection='3d')
        scatter_plot = ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2],
                               c=(self.points[:, 0] + self.points[:, 1] + self.points[:, 2]), cmap="spring")
        plt.colorbar(scatter_plot)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.title("Point cloud")

    def phase_tilts(self):
        # Generate phase tilts in frequency space

        xyrange = self.Nn / 2 * self.dxn
        dkxy = np.pi / xyrange
        kxy = np.arange(-self.Nn / 2 * dkxy, (self.Nn / 2) * dkxy, dkxy)
        dkz = np.pi / self.zrange
        kz = np.arange(-self.Nzn / 2 * dkz, (self.Nzn / 2) * dkz, dkz)

        if self.use_cupy:
            self.phasetilts = cp.zeros((self.n_PhaseStep, self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
        else:
            self.phasetilts = np.zeros((self.n_PhaseStep, self.Nzn, self.Nn, self.Nn), dtype=np.complex64)

        print("Calculating pointwise phase tilts")

        start_time = time.time()

        for xx in range(self.n_PhaseStep):
            if self.use_cupy:
                pxyz = cp.zeros((self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
            else:
                pxyz = np.zeros((self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
            for i in range(self.npoints):
                x = self.points[i, 0]
                y = self.points[i, 1]
                z = self.points[i, 2] + self.dz / self.n_PhaseStep * (xx - 1)
                ph = self.eta * 4 * np.pi * self.NA / self.wavelength
                p1 = -xx * 2 * np.pi / self.n_PhaseStep
                # p1 = -xx * 2 * np.pi / 7
                p2 = xx * 4 * np.pi / self.n_PhaseStep
                # p2 = xx * 4 * np.pi / 7
                if self.axial:  # axial polarisation normalised to peak intensity of 1
                    ill = 2 / 9 * (3 / 2 + np.cos(ph * y + p1 - p2) + np.cos(ph * (y - np.sqrt(3) * x) / 2 + p1) +
                                   np.cos(ph * (-y - np.sqrt(3) * x) / 2 + p2))
                else:  # in plane polarisation normalised to peak intensity of 1
                    ill = 2 / 9 * (3 - np.cos(ph * (-x - y) / 2 + p1 - p2) - np.cos(ph * (- x) + p1)
                                   - np.cos(ph * (y - x) / 2 + p2))
                    # ill = 2 / 9 * (3 - np.cos(ph * y + p1 - p2) - np.cos(ph * (y - np.sqrt(3) * x) / 2 + p1)
                    # - np.cos(ph * (-y - np.sqrt(3) * x) / 2 + p2))
                if self.use_cupy:
                    px = cp.array(np.exp(1j * np.single(x * kxy)))
                    py = cp.array(np.exp(1j * np.single(y * kxy)))
                    pz = cp.array(np.exp(1j * np.single(z * kz)) * ill)
                    pxy = cp.array(px[:, np.newaxis] * py)
                    for ii in range(len(kz)):
                        pxyz[ii, :, :] = pxy * pz[ii]
                    self.phasetilts[xx, :, :, :] = self.phasetilts[xx, :, :, :] + pxyz
                else:
                    px = np.exp(1j * np.single(x * kxy))
                    py = np.exp(1j * np.single(y * kxy))
                    pz = np.exp(1j * np.single(z * kz)) * ill
                    pxy = px[:, np.newaxis] * py
                    for ii in range(len(kz)):
                        pxyz[ii, :, :] = pxy * pz[ii]
                    self.phasetilts[xx, :, :, :] = self.phasetilts[xx, :, :, :] + pxyz

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
            img = cp.zeros((self.Nz * self.n_PhaseStep, self.N, self.N), dtype=np.single)
        else:
            img = np.zeros((self.Nz * self.n_PhaseStep, self.N, self.N), dtype=np.single)
        for i in range(self.n_PhaseStep):
            if self.use_cupy:
                ootf = cp.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                img[cp.arange(i, self.Nz * self.n_PhaseStep, self.n_PhaseStep), :, :] = cp.abs(cp.fft.ifftn(ootf, (self.Nz, self.N, self.N)))
            else:
                ootf = np.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                img[np.arange(i, self.Nz * self.n_PhaseStep, self.n_PhaseStep), :, :] = np.abs(np.fft.ifftn(ootf, (self.Nz, self.N, self.N)))
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


if __name__ == '__main__':
    t = HexSIM_simulator(None, 0, 1, 512, 5.5, 60, 1.1, 1.33, 0.52, 500, 7.0,
                 0.4, 3.0)
    # t.point_cloud()
    # t.phase_tilts()
    t.raw_image_stack()
