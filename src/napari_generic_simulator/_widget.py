"""
Widget of the napari plugin
"""
__author__ = "Meizhu Liang @Imperial College London"

from magicgui import magicgui
from enum import Enum
from .baseSIMulator import import_cp, import_torch, torch_GPU
from .hexSIMulator import HexSim_simulator, RightHexSim_simulator
from .conSIMulator import ConSim_simulator
from .Illumination import ConIll, HexIll, RaHexIll
from qtpy.QtWidgets import QWidget, QVBoxLayout, QFileDialog
from napari.qt.threading import thread_worker
from magicgui.widgets import SpinBox, Label, Container, ComboBox, FloatSpinBox, LineEdit, RadioButtons, PushButton
from napari.layers import Layer
import tifffile
import numpy as np
import open3d as o3d


class Samples(Enum):
    SPHEROIDAL = 500
    FILAMENTS = 10


class PointCloud(QWidget):
    """A widget to go with the SIMulator widget that could generate, display, load and save point clouds."""

    def __init__(self, viewer: 'napari.viewer.Viewer'):
        self._viewer = viewer
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Sets up the layout and adds the widget to the ui"""
        self.cloud_widgets()
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.add_magic_function(self.c_w, layout)

    def add_magic_function(self, function, _layout):
        """Adds the widget to the viewer"""
        self._viewer.layers.events.inserted.connect(function.reset_choices)
        self._viewer.layers.events.removed.connect(function.reset_choices)
        _layout.addWidget(function.native)

    def point_cloud_fil(self, nSamples, L, dL):
        """Generates a point-cloud as the object in the imaging system."""

        # L: full length of each filament in um
        # dL: target length of each short bit to form a filament

        for i in range(nSamples):
            alpha = 2 * np.pi * np.random.rand()
            beta = np.pi / 4 * (1 - 2 * np.random.rand())

            # centres of filaments
            xS = -L / 2 + L * np.random.rand()
            yS = -L / 2 + L * np.random.rand()
            zS = 0

            # starting points of filaments
            x1 = xS + (L / 2 * np.cos(alpha) * np.cos(beta))
            y1 = yS + (L / 2 * np.sin(alpha) * np.cos(beta))
            z1 = zS + (L / 2 * np.sin(beta))

            # ending points of filaments
            x2 = xS - (L / 2 * np.cos(alpha) * np.cos(beta))
            y2 = yS - (L / 2 * np.sin(alpha) * np.cos(beta))
            z2 = zS - (L / 2 * np.sin(beta))

            px = 2 * np.pi * np.random.rand()
            fx = 2 * np.pi * 0.5 * np.random.rand()
            ax = L / 10 * np.random.rand()
            py = 2 * np.pi * np.random.rand()
            fy = 2 * np.pi * 1.0 * np.random.rand()
            ay = L / 10 * np.random.rand()
            pz = 2 * np.pi * np.random.rand()
            fz = 2 * np.pi * 1.0 * np.random.rand()
            az = L / 10 * np.random.rand()

            # adding some perturbation
            l = np.arange(0, 1, dL / L)
            x = x1 + (x2 - x1) * l + ax * np.cos(fx * l + px)
            y = y1 + (y2 - y1) * l + ay * np.cos(fy * l + py)
            z = z1 + (z2 - z1) * l + az * np.cos(fz * l + pz)

            xd = x[1:] - x[:-1]
            yd = y[1:] - y[:-1]
            zd = z[1:] - z[:-1]

            # steps = np.sqrt(xd ** 2 + yd ** 2 + zd ** 2)
            # print(f'length = {np.sum(steps):.2f} um, average = {1000 * np.mean(steps) :.2f} nm, stdev = {1000 * np.std(steps) :.2f} nm')
            if i == 0:
                points = np.single(np.dstack([x, y, z]).squeeze())
            else:
                points = np.concatenate((points, np.single(np.dstack([x, y, z]).squeeze())))
        return points

    def point_cloud(self, npoints, rad, dep):
        """Generates a point - cloud as the object in the imaging system."""

        # rad: radius of sphere of points
        # Multiply the points several times to get the enough number
        pointsxn = (2 * np.random.rand(npoints * 3, 3) - 1) * [rad, rad, rad]

        pointsxnr = np.sum(pointsxn * pointsxn, axis=1)  # multiple times the points
        points_sphere = pointsxn[pointsxnr < (rad ** 2), :]  # simulate spheres from cubes
        points = np.single(points_sphere[(range(npoints)), :])
        points[:, 2] = dep / rad * points[:, 2]  # to make the point cloud for OTF a ellipsoid rather than a sphere
        return points

    def gen_pc(self):
        if self.w_samples.value == Samples.FILAMENTS:
            self.pc = self.point_cloud_fil(self.fil_n.value, self.fil_len.value, self.fil_step.value)
        elif self.w_samples.value == Samples.SPHEROIDAL:
            self.pc = self.point_cloud(self.sph_points.value, self.sph_rad.value, self.sph_dep.value)
        print('Point cloud generated')
        if hasattr(self, 'pc'):
            try:
                self._viewer.add_points(-self.pc[:, ::-1], size=self.fil_step.value, name=self.w_samples.value)
            except Exception as e:
                print(e)

    def save_pc(self):
        if hasattr(self._viewer.layers.selection.active, 'data'):
            try:
                options = QFileDialog.Options()
                filename = QFileDialog.getSaveFileName(self, 'Save a file', options=options, filter='Files (*.pcd)')
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(-self._viewer.layers.selection.active.data[:, ::-1])
                o3d.io.write_point_cloud(filename[0], pcd)
                print('Point cloud saved')
            except Exception as e:
                print(e)

    def load_pc(self):
        try:
            options = QFileDialog.Options()
            filename = QFileDialog.getOpenFileName(self, 'Pick a file', options=options, filter='Files (*.pcd)')
            pcd = o3d.io.read_point_cloud(filename[0])
            out_arr = np.asarray(pcd.points)
            self._viewer.add_points(out_arr, size=0.1, name=filename[0])
        except Exception as e:
            print(e)

    def cloud_widgets(self):
        """Creates a widget containing all small widgets"""
        self.w_samples = RadioButtons(value=Samples.SPHEROIDAL, choices=Samples)


        self.sph_points = SpinBox(value=500, step=50, label='spheroidal_points')
        self.sph_rad = SpinBox(value=5, label='spheroidal_radius (μm)')
        self.sph_dep = FloatSpinBox(value=2.5, step=0.5, max=self.sph_rad.value, label='spheroidal_depth (μm)')

        self.w_sph = Container(widgets=[self.sph_points, self.sph_dep, self.sph_rad])

        self.fil_n = SpinBox(value=20, max=100, label='filament_n')
        self.fil_len = SpinBox(value=5, label='filament_length (μm)')
        self.fil_step = FloatSpinBox(value=0.05, step=0.01, max=self.fil_len.value, label='filament_step (μm)')
        self.w_fil = Container(widgets=[self.fil_n, self.fil_len, self.fil_step])

        self.comprehensive_w = Container(widgets=[magicgui(self.save_pc, call_button='Save current layer as .pcd',
                                                           auto_call=False),
                                                  magicgui(self.load_pc, call_button='Load point cloud',
                                                           auto_call=False)], layout="horizontal", labels=None)
        self.c_w = Container(widgets=[self.w_samples, self.w_sph, self.w_fil, magicgui(self.gen_pc,
                                                                                       call_button='Generate point cloud',
                                                                                       auto_call=False),
                                      self.comprehensive_w], labels=None)


class Sim_mode(Enum):
    HEXSIM = 0
    HEXSIM_RA = 1
    SIM_CONV = 2  # Conventional 2-beam SIM


class Pol(Enum):
    azimuthal = 0  # in_plane
    radial = 1  # axial
    circular = 2
    horizontal = 3
    vertical = 4


class Psf_calc(Enum):
    SCALAR = 0
    VEC_FLEXI = 1
    VEC_RIGID = 2


class Accel(Enum):
    NUMPY = 0
    if import_cp:
        CUPY = 3
    if import_torch:
        TORCH_CPU = 1
        if torch_GPU:
            TORCH_GPU = 2



class SIMulator(QWidget):
    """
    A Napari plugin for the simulation of raw images produced while scanning an object (3D point cloud) through focus as
    the hexSIM illumination pattern is shifted through 7 positions laterally.The raw data is processed by a standard, a
    frame-by-frame and a batch reconstruction to produce the super-resolved output.
    https://doi.org/10.1098/rsta.2020.0162
    This currently supports hexagonal SIM (1 angle, 7 phases) with three beams and that at right-angles.
    """

    def __init__(self, viewer: 'napari.viewer.Viewer'):
        self._viewer = viewer
        super().__init__()
        self.parameters()
        self.setup_ui()
        self.start_simulator()

    def setup_ui(self):
        """Sets up the layout and adds the widget to the ui"""
        self.wrap_widgets()
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.add_magic_function(self.w, layout)

    def add_magic_function(self, function, _layout):
        """Adds the widget to the viewer"""
        self._viewer.layers.events.inserted.connect(function.reset_choices)
        self._viewer.layers.events.removed.connect(function.reset_choices)
        _layout.addWidget(function.native)

    def parameters(self):
        self.SIM_mode = ComboBox(value=Sim_mode.SIM_CONV, label='SIM_mode', choices=Sim_mode)
        self.Polarisation = ComboBox(value=Pol.radial, label='Polarisation', choices=Pol)
        self.Acceleration = ComboBox(value=list(Accel)[-1], label='Acceleration', choices=Accel)
        self.Psf = ComboBox(value=Psf_calc.SCALAR, label='Psf calculation', choices=Psf_calc)
        self.N = SpinBox(value=128, name='spin', label='N pixel')
        self.pixel_size = FloatSpinBox(value=6.5, name='spin', label='pixel size(μm)', step=0.5)
        self.magnification = SpinBox(value=60, name='spin', label='magnification')
        self.ill_NA = FloatSpinBox(value=1, name='spin', label='NA  illumination', min=0.0, step=0.1)
        self.det_NA = FloatSpinBox(value=1.0, name='spin', label='NA  detection', min=0.0, step=0.1)
        self.n = FloatSpinBox(value=1.33, name='spin', label='n', min=0.00)
        self.ill_wavelength = SpinBox(value=540, label='λ  illumination(nm)', step=50)
        self.det_wavelength = SpinBox(value=540, label='λ  detection(nm)', step=50)

        self.zrange = FloatSpinBox(value=3.5, name='spin', label='z range(μm)', min=0.0)
        self.tpoints = FloatSpinBox(value=140, name='spin', label='tpoints', min=0, max=500, step=1)
        self.xdrift = FloatSpinBox(value=0.0, name='spin', label='xdrift(nm)', min=0.0, max=1000.0, step=5)
        self.zdrift = FloatSpinBox(value=50.0, name='spin', label='zdrift(nm)', min=0.0, max=1000.0, step=5)
        self.fwhmz = FloatSpinBox(value=3.0, name='spin', label='fwhmz(μm)', min=0.0, max=10.0)
        self.random_seed = SpinBox(value=123, name='spin', label='random seed')
        self.drift = FloatSpinBox(value=0.0, name='spin', label='drift(nm)', min=0.0, max=1000.0, step=5)
        self.defocus = FloatSpinBox(value=0.0, name='spin', label='de-focus(μm)', min=-10.0, max=10, step=5)
        self.sph_abb = FloatSpinBox(value=0.0, name='spin', label='spherical(rad)', min=-10.0, max=10, step=0.5)
        self.lable = Label(value='aberration')

    def par_list(self):
        """return the current parameter list"""
        return [self.SIM_mode.value, self.Polarisation.value, self.Acceleration.value,
                self.Psf.value, self.N.value, self.pixel_size.value, self.magnification.value, self.ill_NA.value,
                self.det_NA.value, self.n.value, self.ill_wavelength.value, self.det_wavelength.value,
                self.zrange.value, self.tpoints.value, self.xdrift.value, self.zdrift.value,
                self.fwhmz.value, self.random_seed.value, self.drift.value, self.defocus.value, self.sph_abb.value]

    def set_att(self):
        """Sets attributes in the simulation class. Executed frequently to update the parameters"""
        if self.SIM_mode.value == Sim_mode.HEXSIM:
            # self.sim = HexSim_simulator()
            self.sim = HexIll()
            nsteps = self.sim._phaseStep * self.sim._angleStep
        elif self.SIM_mode.value == Sim_mode.HEXSIM_RA:
            self.sim = RaHexIll()
            nsteps = self.sim._phaseStep * self.sim._angleStep
        elif self.SIM_mode.value == Sim_mode.SIM_CONV:
            # self.sim = ConSim_simulator()
            self.sim = ConIll()
            nsteps = self.sim._phaseStep * self.sim._angleStep

        if self.Polarisation.value == Pol.azimuthal:
            self.sim.pol = 'a'
        elif self.Polarisation.value == Pol.radial:
            self.sim.pol = 'r'
        elif self.Polarisation.value == Pol.circular:
            self.sim.pol = 'c'
        elif self.Polarisation.value == Pol.horizontal:
            self.sim.pol = 'h'
        elif self.Polarisation.value == Pol.vertical:
            self.sim.pol = 'v'

        if self.Acceleration.value == Accel.NUMPY:
            self.sim.acc = 0

        if hasattr(Accel, 'TORCH_CPU'):
            if self.Acceleration.value == Accel.TORCH_CPU:
                self.sim.acc = 1
        if hasattr(Accel, 'TORCH_GPU'):
            if self.Acceleration.value == Accel.TORCH_GPU:
                self.sim.acc = 2
        if hasattr(Accel, 'CUPY'):
            if self.Acceleration.value == Accel.CUPY:
                self.sim.acc = 3

        if self.Psf.value == Psf_calc.VEC_RIGID:
            self.sim.psf_calc = 'vector_rigid'
        elif self.Psf.value == Psf_calc.VEC_FLEXI:
                self.sim.psf_calc = 'vector_flexible'
        elif self.Psf.value == Psf_calc.SCALAR:
            self.sim.psf_calc = 'scalar'

        self.sim.points = self.points
        self.sim.npoints = self.npoints

        self.sim.N = self.N.value
        self.sim.pixel_size = self.pixel_size.value
        self.sim.magnification = self.magnification.value
        self.sim.ill_NA = self.ill_NA.value
        self.sim.det_NA = self.det_NA.value
        self.sim.n = self.n.value
        self.sim.ill_wavelength = self.ill_wavelength.value * 1e-3
        self.sim.det_wavelength = self.det_wavelength.value * 1e-3
        self.sim.zrange = self.zrange.value
        self.sim.tpoints = (self.tpoints.value // nsteps // 2) * nsteps * 2
        self.tpoints.value = self.sim.tpoints
        self.sim.dz = 2 * self.zrange.value * nsteps / self.sim.tpoints
        self.sim.xdrift = self.xdrift.value
        self.sim.zdrift = self.zdrift.value
        self.sim.fwhmz = self.fwhmz.value
        self.sim.drift = self.drift.value
        self.sim.random_seed = self.random_seed.value
        self.sim.defocus = self.defocus.value
        self.sim.sph_abb = self.sph_abb.value
        self.used_par_list = [self.SIM_mode.value, self.Polarisation.value, self.Acceleration.value,
                              self.Psf.value,
                              self.N.value, self.pixel_size.value, self.magnification.value, self.ill_NA.value,
                              self.det_NA.value,
                              self.n.value, self.ill_wavelength.value, self.det_wavelength.value,
                              self.zrange.value, self.tpoints.value, self.xdrift.value, self.zdrift.value,
                              self.fwhmz.value, self.random_seed.value, self.drift.value, self.defocus.value,
                              self.sph_abb.value]

    def start_simulator(self):
        """Starts the raw images generators and create the frequency space"""
        if hasattr(self, 'sim'):
            self.stop_simulator()
            self.start_simulator()
        else:
            pass

    def stop_simulator(self):
        if hasattr(self, 'sim'):
            delattr(self, 'sim')

    def select_layer(self, layer: Layer):
        """
        Selects a layer used to simulate raw SIM stacks, it contains the raw point-cloud data.
        Layer : napari.layers.Image
        """
        if not isinstance(layer, Layer):
            return
        if hasattr(self, 'points'):
            delattr(self, 'points')
        self.points = -layer.data[:, ::-1]
        self.npoints = self.points.shape[0]
        self.messageBox.value = f'Selected sample layer: {layer.name}'
        self.get_results()

    def get_results(self):
        if not hasattr(self, 'points'):
            self.messageBox.value = f'Please select a point-cloud layer'
        else:
            def show_img(data):
                self._viewer.add_image(data, name='raw image stack',
                                       scale=(self.zdrift.value * 0.001,
                                              self.pixel_size.value / self.magnification.value,
                                              self.pixel_size.value / self.magnification.value),
                                       translate=(-self.zdrift.value * 0.001 * self.tpoints.value / 2,
                                                  -self.pixel_size.value / self.magnification.value * self.N.value / 2,
                                                  -self.pixel_size.value / self.magnification.value * self.N.value / 2),
                                       metadata={'mode': str(self.SIM_mode.value),
                                                 'pol': str(self.Polarisation.value),
                                                 'acc': str(self.Acceleration.value), 'psf': str(self.Psf.value),
                                                 'N': self.N.value, 'pix size': self.pixel_size.value,
                                                 'mag': self.magnification.value, 'ill NA': self.ill_NA.value,
                                                 'det NA': self.det_NA.value, 'n': self.n.value,
                                                 'ill_wavelength': self.ill_wavelength.value,
                                                 'det_wavelength': self.det_wavelength.value,
                                                 'z range': self.zrange.value,
                                                 'tpoints': self.tpoints.value, 'xdrift': self.xdrift.value,
                                                 'zdrift': self.zdrift.value, 'fwhmz': self.fwhmz.value,
                                                 'random seed': self.random_seed.value, 'Brownian': self.drift.value,
                                                 'defocus': self.defocus.value, 'sph_abb': self.sph_abb.value
                                                 })
                current_step = list(self._viewer.dims.current_step)
                for dim_idx in [-3, -2, -1]:
                    current_step[dim_idx] = data.shape[dim_idx] // 2
                self._viewer.dims.current_step = current_step
                delattr(self, 'points')

            @thread_worker(connect={"returned": show_img})
            def _get_results():
                self.set_att()
                # if self.sim.drift != 0.0:
                t = self.sim.raw_image_stack_brownian()
                # else:
                #     t = self.sim.raw_image_stack()
                try:
                    while True:
                        self.messageBox.value = next(t)
                except Exception as e:
                    print(e)
                return self.sim.img

            _get_results()

    def wrap_widgets(self):
        """Creates a widget containing all small widgets"""
        w_parameters = Container(
            widgets=[
                Container(widgets=[self.SIM_mode, self.Polarisation, self.Acceleration, self.Psf, self.N,
                                   self.pixel_size, self.ill_NA, self.det_NA, self.n,
                                   self.ill_wavelength, self.det_wavelength]),
                Container(widgets=[self.magnification, self.zrange, self.tpoints, self.xdrift,
                                   self.zdrift, self.fwhmz, self.random_seed, self.drift, self.defocus,
                                   self.sph_abb])], layout='horizontal')
        w_cal = magicgui(self.get_results, call_button='Calculate raw image stack', auto_call=False)

        # 'save and print' widgets
        save_tif_with_tags = PushButton(text='save_tif_with_tags')

        @save_tif_with_tags.clicked.connect
        def on_save_tif_with_tags_click():
            """Saves the selected image layer as a tif file with tags"""
            if hasattr(self._viewer.layers.selection.active, 'data'):
                try:
                    options = QFileDialog.Options()
                    filename = QFileDialog.getSaveFileName(self, "Save a file", options=options,
                                                           filter='Images (*.tif)')
                    tifffile.imwrite(filename[0], self._viewer.layers.selection.active.data,
                                     description=str(self._viewer.layers.selection.active.metadata))
                except Exception as e:
                    print(str(e))

        show_psf = PushButton(text='show_psf')

        @show_psf.clicked.connect
        def on_show_psf_click():
            if hasattr(self, 'sim'):
                if hasattr(self.sim, 'psf'):
                    if self.used_par_list != self.par_list():
                        self.messageBox.value = 'Parameters changed! Calculate the raw-image stack first!'
                    else:
                        try:
                            self._viewer.add_image(self.sim.psf,
                                                   scale=(self.sim.dzn,
                                                          self.sim.dxn,
                                                          self.sim.dxn),
                                                   translate=(
                                                       -self.sim.dzn * self.sim.Nzn / 2,
                                                       -self.sim.dxn * self.sim.Nn / 2,
                                                       -self.sim.dxn * self.sim.Nn / 2),
                                                   name='3d-PSF')
                        except Exception as e:
                            print(e)

        print_tif = PushButton(text='print_tags')

        @print_tif.clicked.connect
        def on_print_tif_click():
            """Prints tags of the selected tif image"""
            try:
                if self._viewer.layers.selection.active.source.path is None:
                    print(self._viewer.layers.selection.active.metadata)
                else:
                    frames = tifffile.TiffFile(self._viewer.layers.selection.active.source.path)
                    page = frames.pages[0]
                    # Print file description
                    print(f'==={self._viewer.layers.selection.active.name}.tif===\n' + page.tags["ImageDescription"].value)
                self.messageBox.value = 'Parameters printed'
            except Exception as e:
                print(str(e))

        show_otf = PushButton(text='show_otf')
        @show_otf.clicked.connect
        def on_show_otf_click():
            if hasattr(self, 'sim'):
                if hasattr(self.sim, 'aotf'):
                    if self.used_par_list != self.par_list():
                        self.messageBox.value = 'Parameters changed! Calculate the raw-image stack first!'
                    else:
                        try:
                            self._viewer.add_image(self.sim.aotf,
                                                   scale=(self.sim.dzn,
                                                          self.sim.dxn,
                                                          self.sim.dxn),
                                                   translate=(
                                                       -self.sim.dzn * self.sim.Nzn / 2,
                                                       -self.sim.dxn * self.sim.Nn / 2,
                                                       -self.sim.dxn * self.sim.Nn / 2),
                                                   name='3d-OTF')
                        except Exception as e:
                            print(str(e))

        show_illumination = PushButton(text='illumination')

        @show_illumination.clicked.connect
        def on_show_illumination_click():
            if hasattr(self, 'sim'):
                if self.used_par_list != self.par_list():
                    self.messageBox.value = 'Parameters changed! Calculate the raw-image stack first!'
                else:
                    try:
                        self._viewer.add_image(self.sim.illumination_stack(),
                                               scale=(self.zdrift.value * 0.001,
                                                      self.pixel_size.value / self.magnification.value,
                                                      self.pixel_size.value / self.magnification.value),
                                               translate=(-self.zdrift.value * 0.001 * self.tpoints.value / 2,
                                                          -self.pixel_size.value / self.magnification.value * self.N.value / 2,
                                                          -self.pixel_size.value / self.magnification.value * self.N.value / 2),
                                               interpolation2d='spline36',
                                               name='illumination')
                    except Exception as e:
                        print(str(e))

        w_save_and_print = Container(widgets=[save_tif_with_tags, print_tif], layout='horizontal')
        w_show = Container(widgets=[show_psf, show_otf, show_illumination], layout='horizontal')
        self.messageBox = LineEdit(value='Messages')
        self.w = Container(widgets=[w_parameters, magicgui(self.select_layer, call_button='Calculate results'),
                                    w_save_and_print, w_show, self.messageBox],
                           labels=None)
