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
    random_seed = None


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
        self._tdev = torch.device('cuda' if self.acc == 3 else 'cpu')

    def point_cloud(self):

        rad = 10  # radius of sphere of points
        # Multiply the points several times to get the enough number
        pointsxn = (2 * np.random.rand(self.npoints * 3, 3) - 1) * [rad, rad, rad]

        pointsxnr = np.sum(pointsxn * pointsxn, axis=1)  # multiple times the points
        points_sphere = pointsxn[pointsxnr < (rad ** 2), :]  # simulate spheres from cubes
        self.points = points_sphere[(range(self.npoints)), :]
        self.points[:, 2] = self.points[:, 2] / 2  # to make the point cloud for OTF a ellipsoid rather than a sphere

    def phase_tilts(self):
        """Generate phase tilts in frequency space"""
        self._nsteps = self._phaseStep * self._angleStep
        xyrange = self.Nn / 2 * self.dxn
        dkxy = np.pi / xyrange
        dkz = np.pi / self.zrange
        self.kxy = np.arange(-self.Nn / 2 * dkxy, (self.Nn / 2) * dkxy, dkxy)
        self.kz = np.arange(-self.Nzn / 2 * dkz, (self.Nzn / 2) * dkz, dkz)

        if self.acc == 0:
            self.phasetilts = np.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
        elif self.acc == 1:
            self.phasetilts = cp.zeros((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=np.complex64)
        else:
            self.phasetilts = torch.empty((self._nsteps, self.Nzn, self.Nn, self.Nn), dtype=torch.complex64, device=self._tdev)

        start_time = time.time()

        itcount = 0
        total_its = self._angleStep * self._phaseStep * self.npoints
        lastProg = -1

        for astep in range(self._angleStep):
            for pstep in range(self._phaseStep):
                for i in range(self.npoints):
                    prog = (100 * itcount) // total_its
                    if prog > lastProg:
                        lastProg = prog
                        yield f'Phase tilts calculation: {prog:.1f}% done'
                    itcount += 1
                    isteps = pstep + self._angleStep * astep  # index of the steps
                    self.x = self.points[i, 0]
                    self.y = self.points[i, 1]
                    z = self.points[i, 2] + self.dz / self._nsteps * (isteps)
                    self.ph = self.eta * 4 * np.pi * self.NA / self.wavelength
                    self.p1 = pstep * 2 * np.pi / self._phaseStep
                    self.p2 = -pstep * 4 * np.pi / self._phaseStep
                    self._ill()  # gets illumination from the child class
                    if self.pol == 1:
                        ill = self._illAx
                    elif self.pol == 2:
                        ill = self._illCi
                    else:
                        ill = self._illIp
                    if self.acc == 0:
                        px = np.exp(1j * np.single(self.x * self.kxy))[:, np.newaxis]
                        py = np.exp(1j * np.single(self.y * self.kxy))
                        pz = (np.exp(1j * np.single(z * self.kz)) * ill)[:, np.newaxis, np.newaxis]
                        self.phasetilts[isteps, :, :, :] += (px * py) * pz
                    elif self.acc == 1:
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

    def raw_image_stack(self):
        # Calculates point cloud, phase tilts, 3d psf and otf before the image stack
        self.initialise()
        self.point_cloud()
        yield "Point cloud calculated"

        for msg in self.phase_tilts():
            yield msg

        # Calculating psf
        nz = 0
        psf = np.zeros((self.Nzn, self.Nn, self.Nn))
        pupil = self.kr < 1
        for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
            c = (np.exp(
                1j * (z * self.n * 2 * np.pi / self.wavelength *
                      np.sqrt(1 - (self.kr * pupil) ** 2 * self.NA ** 2 / self.n ** 2)))) * pupil
            psf[nz, :, :] = abs(np.fft.fftshift(np.fft.ifft2(c))) ** 2 * np.exp(-z ** 2 / 2 / self.sigmaz ** 2)
            nz = nz + 1
        # Normalised so power in resampled psf(see later on) is unity in focal plane
        psf = psf * self.Nn ** 2 / np.sum(pupil) * self.Nz / self.Nzn
        self.psf_z0 = psf[int(self.Nzn / 2 + 10), :, :]  # psf at z=0
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
        elif self.acc == 1:
            img = cp.zeros((self.Nz * self._nsteps, self.N, self.N), dtype=np.single)
        else:
            img =torch.empty((self.Nz * self._nsteps, self.N, self.N), dtype=torch.float, device=self._tdev)

        for i in range(self._nsteps):
            if self.acc == 0:
                ootf = np.fft.fftshift(otf) * self.phasetilts[i, :, :, :]
                img[np.arange(i, self.Nz * self._nsteps, self._nsteps), :, :] = abs(
                    np.fft.ifftn(ootf, (self.Nz, self.N, self.N)))
            elif self.acc == 1:
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
            self.img_sum_z = cp.asnumpy(cp.sum(img, axis=0))
            self.img_sum_x = cp.asnumpy(cp.sum(img, axis=1))
            self.img = cp.asnumpy(img)
        elif self.acc == 2:
            self.img_sum_z = (torch.sum(img, axis=0)).numpy()
            self.img_sum_x = (torch.sum(img, axis=1)).numpy()
            self.img = img.numpy()
        elif self.acc == 3:
            self.img_sum_z = (torch.sum(img, axis=0)).detach().cpu().numpy()
            self.img_sum_x = (torch.sum(img, axis=1)).detach().cpu().numpy()
            self.img = img.detach().cpu().numpy()

        # Save generated images
        tifffile.imwrite(stackfilename, self.img)
        print('Raw image stack saved')

        yield f'Finished, Phase tilts calculation time:  {self.elapsed_time:3f}s'
        print(f'Finished, Phase tilts calculation time:  {self.elapsed_time:3f}s')
