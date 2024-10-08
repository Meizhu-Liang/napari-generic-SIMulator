"""
The parent class to simulate raw data for SIM.
Some calculations are adapted by the work by Mark Neil @Imperial College London
"""
__author__ = "Meizhu Liang @Imperial College London"

import numpy as np
import opt_einsum as oe
import time

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
    acc = None  # acceleration
    pol = None  # polarisation
    psf_calc = None
    _tdev = None
    N = 512  # points to use in FFT
    pixel_size = 5.5  # camera pixel size
    magnification = 60  # objective magnification
    ill_NA = 0.75  # numerical aperture at illumination beams
    det_NA = 1.1  # numerical aperture at sample
    n = 1.33  # refractive index at sample
    ill_wavelength = 520e-3  # illumination wavelength in um
    det_wavelength = 570e-3  # detection wavelength in um
    zrangeN = 6  # Nyquist distance either side of focus to calculate in microns (depth of the sample)
    # fwhmz = 3.0  # FWHM of light sheet in z
    random_seed = 123
    drift = 0
    spherical = 0  # spherical aberration in um
    sph_abb = 0
    add_error = False

    def initialise(self):
        if self.acc == 3:
            self.xp = cp
        np.random.seed(self.random_seed)
        # self.seed(1234)  # set random number generator seed
        # self.sigmaz = self.fwhmz / 2.355
        self.dx = self.pixel_size / self.magnification  # Sampling in lateral plane at the sample in um
        self.res = self.det_wavelength / (2 * self.det_NA)
        self.dxn = self.res / 2  # 2 * Nyquist frequency in x and y.
        self.Nn = int(np.ceil(self.N * self.dx / self.dxn / 2) * 2)  # Number of points at Nyquist sampling, even number
        self.dxn = self.N * self.dx / self.Nn  # correct spacing
        oversampling = self.res / self.dxn  # factor by which pupil plane oversamples the coherent psf data
        self.k0_ill = 2 * np.pi * self.n / self.ill_wavelength
        self.k0_det = 2 * np.pi * self.n / self.det_wavelength
        self.krmax = self.det_NA * self.k0_det / self.n
        self.dk = oversampling / (self.Nn / 2) * self.krmax  # Pupil plane sampling
        self.kx, self.ky = self.xp.meshgrid(
            self.xp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn),
            self.xp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn))
        self.kr = np.sqrt(self.kx ** 2 + self.ky ** 2)  # Raw pupil function, pupil defined over circle of radius self.krmax.
        self.spherical = self.sph_abb * np.sqrt(5) * (6 * ((self.kr / self.krmax) ** 4 - (self.kr / self.krmax) ** 2) + 1)
        # Nyquist sampling in z
        self.dzn = self.det_wavelength / (2 * self.n * (1 - np.cos(np.arcsin(self.det_NA / self.n))))
        self.Nzn = int(2 * np.ceil(self.zrangeN / self.dzn))
        self.dzn = 2 * self.zrangeN / self.Nzn  # step size in axial direction of PSF in um
        if (self.acc == 1) | (self.acc == 2):
            self._tdev = torch.device('cuda' if self.acc == 2 else 'cpu')
        dkz = np.pi / self.zrangeN
        if (self.acc == 0) or (self.acc == 3):
            self.kxy = self.dk * self.xp.arange(-self.Nn // 2, self.Nn // 2, 1, dtype=self.xp.single)
            self.kz = dkz * self.xp.arange(-self.Nzn // 2, self.Nzn // 2, 1, dtype=self.xp.single)
            if self.psf_calc == 'vector_rigid':
                # pre allocate phasetilts arrays
                self.phasetilts = [self.xp.zeros((3, self.Nzn, self.Nn, self.Nn), dtype=self.xp.complex64)
                                   for i in range(self._nsteps)]
            else:
                self.phasetilts = [self.xp.zeros((self.Nzn, self.Nn, self.Nn), dtype=self.xp.complex64)
                                   for i in range(self._nsteps)]
        else:
            self.kxy = self.dk * torch.arange(-self.Nn // 2, self.Nn // 2, 1,
                               dtype=torch.float32, device=self._tdev)
            self.kz = dkz * torch.arange(-self.Nzn // 2, self.Nzn // 2, 1,
                              dtype=torch.float32, device=self._tdev)
            if self.psf_calc == 'vector_rigid':
                self.phasetilts = [torch.zeros(3, self.Nzn, self.Nn, self.Nn, dtype=torch.complex64,
                                               device=self._tdev) for i in range(self._nsteps)]
            else:
                self.phasetilts = [torch.zeros(self.Nzn, self.Nn, self.Nn, dtype=torch.complex64,
                                               device=self._tdev) for i in range(self._nsteps)]

    def phase_tilts(self):
        """Generates phase tilts in frequency space"""
        start_time = time.time()
        itcount = 0
        total_its = self._angleStep * self._phaseStep * self.npoints
        lastProg = 0

        for astep in range(self._angleStep):
            self.jones_vectors(astep)
            for pstep in range(self._phaseStep):
                self.points += self.drift * np.random.standard_normal(3) / 1000
                self.points[:, 0] += self.xdrift / 1000
                if self.zdrift:
                    self.points[:, 2] += self.zdrift / 1000
                istep = pstep + self._phaseStep * astep  # index of the steps
                prog = (100 * itcount) // total_its
                if prog > lastProg + 9:
                    lastProg = prog
                    # yield f'Phase tilts calculation: {prog:.1f}% done'
                itcount += 1

                # get illumination from the child class

                if (self.acc == 0) or (self.acc == 3):
                    x = self.xp.array(self.points[:, 0], dtype=self.xp.single)
                    y = self.xp.array(self.points[:, 1], dtype=self.xp.single)
                    z = self.xp.array(self.points[:, 2], dtype=self.xp.single)
                    px = self.xp.exp(1j * self.kxy[self.xp.newaxis, :] * x[:, self.xp.newaxis])
                    py = self.xp.exp(1j * self.kxy[self.xp.newaxis, :] * y[:, self.xp.newaxis])
                    pz = self.xp.exp(1j * self.kz[self.xp.newaxis, :] * z[:, self.xp.newaxis])
                    if self.psf_calc == 'vector_rigid':
                        ill = self.xp.array(self._ill_obj_vec(x, y, z, pstep, astep),
                                            dtype=self.xp.single)
                        oe.contract('im,il,ik,ij->mjkl', ill, px, py, pz, out=self.phasetilts[istep])
                    else:
                        ill = self.xp.array(self._ill_obj(x, y, z, pstep, astep),
                                            dtype=self.xp.single)
                        oe.contract('i,il,ik,ij->jkl', ill, px, py, pz, out=self.phasetilts[istep])
                else:
                    x = torch.tensor(self.points[:, 0], dtype=torch.float32, device=self._tdev)
                    y = torch.tensor(self.points[:, 1], dtype=torch.float32, device=self._tdev)
                    z = torch.tensor(self.points[:, 2], dtype=torch.float32, device=self._tdev)

                    px = torch.exp(1j * self.kxy[None, :] * x[:, None])
                    py = torch.exp(1j * self.kxy[None, :] * y[:, None])
                    pz = torch.exp(1j * self.kz[None, :] * z[:, None])
                    if self.psf_calc == 'vector_rigid':
                        ill = torch.tensor(self._ill_obj_vec(self.points[:, 0], self.points[:, 1], self.points[:, 2], pstep, astep),
                                           dtype=torch.float32, device=self._tdev)
                        oe.contract('im,il,ik,ij->mjkl', ill, px, py, pz, out=self.phasetilts[istep])
                    else:
                        ill = torch.tensor(self._ill_obj(self.points[:, 0], self.points[:, 1], self.points[:, 2], pstep, astep),
                                           dtype=torch.float32, device=self._tdev)
                        oe.contract('i,il,ik,ij->jkl', ill, px, py, pz, out=self.phasetilts[istep])
        self.elapsed_time = time.time() - start_time
        yield f'{self.elapsed_time:.3f}s'

    def get_vector_psf(self):
        # use krmax to define the pupil function
        kx = self.kx + 1e-7  # need to add small offset to avoid division by zero
        ky = self.ky + 1e-7
        kr2 = kx ** 2 + ky ** 2  # square kr
        pupil = 1.0 * (kr2 < self.krmax ** 2)  # change of the pupil in the amplitude
        kz = np.sqrt((self.k0_det ** 2 - kr2) + 0j)

        # Calculating psf
        nz = 0
        psf_x = self.xp.zeros((self.Nzn, self.Nn, self.Nn))
        psf_y = self.xp.zeros((self.Nzn, self.Nn, self.Nn))
        psf_z = self.xp.zeros((self.Nzn, self.Nn, self.Nn))

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

        # p2 are the vertices of the same dodecahedron in a different orientation
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
        # got the flattened array and calculate the square root of the sum of squares
        p2 = p2 / self.xp.linalg.norm(p2[0, :])

        p = p2[p2[:, 2] > 0]  # just select one half of the orientations as their emission is symmetric
        plen = p.shape[0]

        s1 = self.xp.array([1, 0, 0])  # x polarised illumination orientation
        excitation1 = (s1 @ p.T) ** 2
        s2 = self.xp.array([0, 1, 0])  # y polarised illumination orientation
        excitation2 = (s2 @ p.T) ** 2
        s3 = self.xp.array([0, 0, 1])  # z polarised illumination orientation
        excitation3 = (s3 @ p.T) ** 2

        fx1 = self.xp.sqrt(self.k0_det / kz) * \
              (self.k0_det * ky ** 2 + kx ** 2 * kz) / (self.k0_det * kr2) * pupil
        fy1 = self.xp.sqrt(self.k0_det / kz) * \
              kx * ky * (kz - self.k0_det) / (self.k0_det * kr2) * pupil
        fx2 = self.xp.sqrt(self.k0_det / kz) * \
              kx * ky * (kz - self.k0_det) / (self.k0_det * kr2) * pupil
        fy2 = self.xp.sqrt(self.k0_det / kz) * \
              (self.k0_det * kx ** 2 + ky ** 2 * kz) / (self.k0_det * kr2) * pupil
        fx3 = self.xp.sqrt(self.k0_det / kz) * kx / self.k0_det * pupil
        fy3 = self.xp.sqrt(self.k0_det / kz) * ky / self.k0_det * pupil

        for z in np.arange(-self.zrangeN, self.zrangeN, self.dzn):
            pupil_phase = np.exp(1j * (z * kz + self.spherical))  # change of the pupil in the phase
            # x-polarised field at camera for x-oriented dipole
            Exx = self.xp.fft.fftshift(
                self.xp.fft.ifft2(fx1 * pupil_phase))
            # y-polarised field at camera for x-oriented dipole
            Exy = self.xp.fft.fftshift(
                self.xp.fft.ifft2(fy1 * pupil_phase))
            # x-polarised field at camera for y-oriented dipole
            Eyx = self.xp.fft.fftshift(
                self.xp.fft.ifft2(fx2 * pupil_phase))
            # y-polarised field at camera for y-oriented dipole
            Eyy = self.xp.fft.fftshift(
                self.xp.fft.ifft2(fy2 * pupil_phase))
            # x-polarised field at camera for z-oriented dipole
            Ezx = self.xp.fft.fftshift(
                self.xp.fft.ifft2(fx3 * pupil_phase))
            # y-polarised field at camera for z-oriented dipole
            Ezy = self.xp.fft.fftshift(
                self.xp.fft.ifft2(fy3 * pupil_phase))
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

            psf_x[nz, :, :] = intensityx
            psf_y[nz, :, :] = intensityy
            psf_z[nz, :, :] = intensityz
            nz = nz + 1
        psf_x = psf_x * self.Nn ** 2 / self.xp.sum(pupil) / plen
        psf_y = psf_y * self.Nn ** 2 / self.xp.sum(pupil) / plen
        psf_z = psf_z * self.Nn ** 2 / self.xp.sum(pupil) / plen
        return psf_x, psf_y, psf_z

    def get_scalar_psf(self):
        # use krmax to define the pupil function
        nz = 0
        psf = self.xp.zeros((self.Nzn, self.Nn, self.Nn))
        pupil = self.kr < (self.krmax)
        kz = np.sqrt(self.k0_det ** 2 - (self.kr * pupil) ** 2)
        for z in np.arange(-self.zrangeN, self.zrangeN,
                           self.dzn):
            c = np.exp(1j * (z * kz + self.spherical)) * pupil
            psf[nz, :, :] = abs(np.fft.fftshift(
                np.fft.ifft2(c, norm='ortho'))) ** 2
            # default 'backwards' normalisation: psf[nz, :, :] = abs(np.fft.fftshift(np.fft.ifft2(c))) ** 2
            nz = nz + 1
        # Normalised so power in resampled psf(see later on) is unity in focal plane
        psf = psf / self.xp.sum(pupil ** 2)
        # normalisation factor: Nn ** 2, for use with norm='backwards' ifft:
        # psf = psf * (self.Nn **2 / self.xp.sum(pupil ** 2))
        return psf

    def raw_image_stack_brownian(self):
        # Calculates 3d psf and otf before the image stack
        self.initialise()
        # Calculating psf
        if self.psf_calc == 'vector_rigid':
            psf_x, psf_y, psf_z = self.get_vector_psf()
            norm_val = np.sum(psf_x, axis=(1, 2)).max()
            psf_x /= norm_val
            psf_y /= norm_val
            psf_z /= norm_val
            psf = psf_z
            self.psf_x = psf_x
            self.psf_y = psf_y
            self.psf_z = psf_z
        elif self.psf_calc == 'vector_flexible':
            psf_x, psf_y, psf_z = self.get_vector_psf()
            psf = psf_x + psf_y + psf_z
            norm_val = np.sum(psf, axis=(1, 2)).max()
            psf /= norm_val
        else:
            psf = self.get_scalar_psf()

        yield "psf calculated"
        self.psffile = np.sum(psf, axis=(1, 2))
        # Calculating 3d otf
        self.psf = psf  # 3d psf for display
        if self.acc == 3:
            self.psf = cp.asnumpy(self.psf)
        psf = self.xp.fft.fftshift(psf, axes=0)  # need to set plane zero as in-focus here
        otf = self.xp.fft.fftn(psf)
        if self.psf_calc == 'vector_rigid':
            otf_x = self.xp.fft.fftn(self.xp.fft.fftshift(psf_x, axes=0))  # need to set plane zero as in-focus here
            otf_y = self.xp.fft.fftn(self.xp.fft.fftshift(psf_y, axes=0))  # need to set plane zero as in-focus here
            otf_z = self.xp.fft.fftn(self.xp.fft.fftshift(psf_z, axes=0))  # need to set plane zero as in-focus here
        aotf = abs(self.xp.fft.fftshift(otf))  # absolute otf, also modulus of otf (MTF)
        self.aotffile = np.sum(aotf, axis=(1, 2))
        if self.acc == 3:
            aotf = cp.asnumpy(aotf)
        m = max(aotf.flatten())
        self.aotf = np.log(aotf + aotf.max() / 1000)  # 3d aotf with log
        if self.acc == 3:
            self.aotf = cp.asnumpy(self.aotf)
        yield "3d otf calculated"

        self.points[:, 0] -= self.xdrift * self.tpoints / 2000
        if self.zdrift:
            self.points[:, 2] -= self.zdrift * self.tpoints / 2000
        else:
            self.points[:, 2] -= self.zstep * (self.tpoints / self._nsteps) / 2000
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
            if self.zstep:
                self.points[:, 2] += self.zstep / 1000
            for i in range(self._nsteps):
                if (self.acc == 0) | (self.acc == 3):
                    if self.psf_calc == 'vector_rigid':
                        ootf = self.xp.fft.fftshift(otf_x) * self.phasetilts[i][0, :, :, :] + \
                               self.xp.fft.fftshift(otf_y) * self.phasetilts[i][1, :, :, :] + \
                               self.xp.fft.fftshift(otf_z) * self.phasetilts[i][2, :, :, :]
                    else:
                        ootf = self.xp.fft.fftshift(otf) * self.phasetilts[i]
                    img[tplane, :, :] = self.xp.abs(
                        self.xp.fft.ifft2(self.xp.sum(ootf, axis=0), (self.N, self.N)))
                else:
                    if self.psf_calc == 'vector_rigid':
                        ootf = torch.fft.fftshift(torch.as_tensor(otf_x, device=self._tdev)) * \
                               self.phasetilts[i][0, :, :, :] + \
                               torch.fft.fftshift(torch.as_tensor(otf_y, device=self._tdev)) * \
                               self.phasetilts[i][1, :, :, :] + \
                               torch.fft.fftshift(torch.as_tensor(otf_z, device=self._tdev)) * \
                               self.phasetilts[i][2, :, :, :]
                    else:
                        ootf = torch.fft.fftshift(torch.as_tensor(otf, device=self._tdev)) * self.phasetilts[i]
                    img[tplane, :, :] = (torch.abs(
                        torch.fft.ifft2(torch.sum(ootf, axis=0), (self.N, self.N)))).to(torch.float)
                tplane += 1

        # OK to use abs here as signal should be all positive.
        # Abs is required as the result will be complex as the fourier plane cannot be shifted back to zero when oversampling.
        # But should reduction in sampling be allowed here(Nz < Nzn)?

        # stackfilename = f"Raw_img_stack_{self.N}_{self.pol}.tif"
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

        # # Save generated images
        # tifffile.imwrite(stackfilename, self.img)
        elapsed_Brownian = time.time() - start_Brownian
        yield f'Finished, Phase tilts calculation time:  {elapsed_Brownian:.3f}s'

    def illumination_stack(self):
        # Calculates 3d psf and otf before the image stack
        print(f'allocating illumination stack: {(int(self.tpoints), self.N, self.N)}')

        illumination = self.xp.zeros((int(self.tpoints), self.N, self.N))
        xyvals = (self.xp.arange(self.N) - self.N / 2) * self.dx
        xarr, yarr = self.xp.meshgrid(xyvals, xyvals)
        xarr_l, yarr_l = self.xp.array(xarr).flatten(), self.xp.array(yarr).flatten()
        zarr_l = self.xp.zeros_like(xarr_l)
        self.npoints = xarr_l.shape[0]

        start_time = time.time()
        itcount = -1
        for astep in range(self._angleStep):
            self.jones_vectors(astep)
            for pstep in range(self._phaseStep):
                itcount += 1
                ill_1d = self._ill_obj(xarr_l, yarr_l, zarr_l, pstep, astep)
                illumination[itcount, :, :] = self.xp.reshape(ill_1d, (self.N, self.N))

        for i in range(1, int(self.tpoints) // self._nsteps):
            for j in range(self._nsteps):
                illumination[i * self._nsteps + j, :, :] = illumination[j, :, :]

        elapsed_time = time.time() - start_time
        print(f'Illumination calculation time:  {elapsed_time:3f}s')
        if self.acc == 3:
            return cp.asnumpy(illumination)
        else:
            return illumination
