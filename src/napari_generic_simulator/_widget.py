"""
Widget of the napari plugin
"""
__author__ = "Meizhu Liang @Imperial College London"

from magicgui import magicgui
from enum import Enum
from .baseSIMulator import import_cp, import_torch, torch_GPU
from .hexSIMulator import HexSim_simulator, RightHexSim_simulator
from .conSIMulator import ConSim_simulator
from qtpy.QtWidgets import QWidget, QVBoxLayout, QFileDialog
from napari.qt.threading import thread_worker
from magicgui.widgets import SpinBox, Label, Container, ComboBox, FloatSpinBox, LineEdit
import tifffile


class Sim_mode(Enum):
    HEXSIM = 0
    HEXSIM_RA = 1
    SIM_CONV = 2  # Conventional 2-beam SIM


class Pol(Enum):
    IN_PLANE = 0
    AXIAL = 1
    CIRCULAR = 2


class Psf_calc(Enum):
    SCALAR = 0
    VECTOR = 1


class Accel(Enum):
    NUMPY = 0
    if import_torch:
        TORCH_CPU = 1
        if torch_GPU:
            TORCH_GPU = 2
    if import_cp:
        CUPY = 3


class Samples(Enum):
    POINTS = 0
    FILAMENTS = 1


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
        self.sample = ComboBox(value=Samples.FILAMENTS, label='Sample', choices=Samples)
        self.SIM_mode = ComboBox(value=Sim_mode.SIM_CONV, label='SIM_mode', choices=Sim_mode)
        self.Polarisation = ComboBox(value=Pol.AXIAL, label='Polarisation', choices=Pol)
        self.Acceleration = ComboBox(value=list(Accel)[-1], label='Acceleration', choices=Accel)
        self.Psf = ComboBox(value=Psf_calc.SCALAR, label='Psf calculation', choices=Psf_calc)
        self.N = SpinBox(value=128, name='spin', label='N pixel')
        self.pixel_size = FloatSpinBox(value=6.5, name='spin', label='pixel size(μm)', step=0.5)
        self.magnification = SpinBox(value=60, name='spin', label='magnification')
        self.ill_NA = FloatSpinBox(value=1.0, name='spin', label='NA  illumination', min=0.0, step=0.1)
        self.det_NA = FloatSpinBox(value=1.0, name='spin', label='NA  detection', min=0.0, step=0.1)
        self.n = FloatSpinBox(value=1.33, name='spin', label='n', min=0.00)
        self.ill_wavelength = SpinBox(value=500, label='λ  illumination(nm)', step=50)
        self.det_wavelength = SpinBox(value=540, label='λ  detection(nm)', step=50)

        self.n_samples = SpinBox(value=20, name='spin', label='N samples', max=10000, step=5)
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
        return [self.sample.value, self.SIM_mode.value, self.Polarisation.value, self.Acceleration.value,
                self.Psf.value, self.N.value, self.pixel_size.value, self.magnification.value, self.ill_NA.value,
                self.det_NA.value, self.n.value, self.ill_wavelength.value, self.det_wavelength.value,
                self.n_samples.value, self.zrange.value, self.tpoints.value, self.xdrift.value, self.zdrift.value,
                self.fwhmz.value, self.random_seed.value, self.drift.value, self.defocus.value, self.sph_abb.value]

    def set_att(self):
        """Sets attributes in the simulation class. Executed frequently to update the parameters"""
        if self.SIM_mode.value == Sim_mode.HEXSIM:
            self.sim = HexSim_simulator()
            nsteps = self.sim._phaseStep * self.sim._angleStep
        elif self.SIM_mode.value == Sim_mode.HEXSIM_RA:
            self.sim = RightHexSim_simulator()
            nsteps = self.sim._phaseStep * self.sim._angleStep
        elif self.SIM_mode.value == Sim_mode.SIM_CONV:
            self.sim = ConSim_simulator()
            nsteps = self.sim._phaseStep * self.sim._angleStep

        if self.sample.value == Samples.POINTS:
            self.sim.sample = 'points'
        else:
            self.sim.sample = 'filaments'

        if self.Polarisation.value == Pol.IN_PLANE:
            self.sim.pol = 'in-plane'
        elif self.Polarisation.value == Pol.AXIAL:
            self.sim.pol = 'axial'
        elif self.Polarisation.value == Pol.CIRCULAR:
            self.sim.pol = 'circular'

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

        if self.Psf.value == Psf_calc.VECTOR:
            self.sim.psf_calc = 'vector'
        elif self.Psf.value == Psf_calc.SCALAR:
            self.sim.psf_calc = 'scalar'

        self.sim.N = self.N.value
        self.sim.pixel_size = self.pixel_size.value
        self.sim.magnification = self.magnification.value
        self.sim.ill_NA = self.ill_NA.value
        self.sim.det_NA = self.det_NA.value
        self.sim.n = self.n.value
        self.sim.ill_wavelength = self.ill_wavelength.value
        self.sim.det_wavelength = self.det_wavelength.value
        self.sim.nSamples = self.n_samples.value
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
        self.used_par_list = [self.sample.value, self.SIM_mode.value, self.Polarisation.value, self.Acceleration.value,
                              self.Psf.value,
                              self.N.value, self.pixel_size.value, self.magnification.value, self.ill_NA.value,
                              self.det_NA.value,
                              self.n.value, self.ill_wavelength.value, self.det_wavelength.value, self.n_samples.value,
                              self.zrange.value, self.tpoints.value, self.xdrift.value, self.zdrift.value,
                              self.fwhmz.value, self.random_seed.value, self.drift.value, self.defocus.value,
                              self.sph_abb.value]

    def start_simulator(self):
        """Starts the raw images generators and create the frequency space"""
        if hasattr(self, 'sim'):
            self.stop_simulator()
            self.start_simulator()
        else:
            self.set_att()

    def stop_simulator(self):
        if hasattr(self, 'sim'):
            delattr(self, 'sim')

    def get_results(self):
        def show_img(data):
            self._viewer.add_image(data, name='raw image stack',
                                   metadata={'sample': str(self.sample.value), 'mode': str(self.SIM_mode.value),
                                             'pol': str(self.Polarisation.value),
                                             'acc': str(self.Acceleration.value), 'psf': str(self.Psf.value),
                                             'N': self.N.value, 'pix size': self.pixel_size.value,
                                             'mag': self.magnification.value, 'ill NA': self.ill_NA.value,
                                             'det NA': self.det_NA.value, 'n': self.n.value,
                                             'ill_wavelength': self.ill_wavelength.value,
                                             'det_wavelength': self.det_wavelength.value,
                                             'n samples': self.n_samples.value, 'z range': self.zrange.value,
                                             'tpoints': self.tpoints.value, 'xdrift': self.xdrift.value,
                                             'zdrift': self.zdrift.value, 'fwhmz': self.fwhmz.value,
                                             'random seed': self.random_seed.value, 'Brownian': self.drift.value,
                                             'defocus': self.defocus.value, 'sph_abb': self.sph_abb.value
                                             })

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

    def show_raw_img_sum(self, show_raw_img_sum: bool = False):
        if show_raw_img_sum:
            if hasattr(self.sim, 'img_sum_z'):
                if self.used_par_list != self.par_list():
                    self.messageBox.value = 'Parameters changed! Calculate the raw-image stack first!'
                else:
                    try:
                        self._viewer.add_image(self.sim.img_sum_z, name='raw image sum along z axis')
                        self._viewer.add_image(self.sim.img_sum_x, name='raw image sum along x (or y) axis')
                    except Exception as e:
                        print(str(e))

    def show_psf(self, show_3D_psf_slice: bool = False):
        if show_3D_psf_slice:
            if hasattr(self.sim, 'psf_z0'):
                if self.used_par_list != self.par_list():
                    self.messageBox.value = 'Parameters changed! Calculate the raw-image stack first!'
                else:
                    try:
                        self._viewer.add_image(self.sim.psf_z0, name='PSF in x-y plane')
                    except Exception as e:
                        print(e)

    def show_otf(self, show_3D_otf_slice: bool = False):
        if show_3D_otf_slice:
            if hasattr(self.sim, 'aotf_x'):
                if self.used_par_list != self.par_list():
                    self.messageBox.value = 'Parameters changed! Calculate the raw-image stack first!'
                else:
                    try:
                        self._viewer.add_image(self.sim.aotf_x, name='OTF perpendicular to x')
                    except Exception as e:
                        print(str(e))

    def save_tiff_with_tags(self):
        if hasattr(self.sim, 'img'):
            try:
                options = QFileDialog.Options()
                filename = QFileDialog.getSaveFileName(self, "Pick a file", options=options, filter="Images (*.tif)")
                tifffile.imwrite(filename[0], self._viewer.layers[self._viewer.layers.selection.active.name].data,
                                 description=str(
                                     self._viewer.layers[self._viewer.layers.selection.active.name].metadata))
            except Exception as e:
                print(str(e))

    def print_tiff_tags(self):
        try:
            frames = tifffile.TiffFile(self._viewer.layers.selection.active.name + '.tif')
            page = frames.pages[0]
            # Print file description
            print(f'==={self._viewer.layers.selection.active.name}.tif===\n' + page.tags["ImageDescription"].value)
            self.messageBox.value = 'Parameters printed'
        except Exception as e:
            print(str(e))

    def wrap_widgets(self):
        """Creates a widget containing all small widgets"""
        w_parameters = Container(
            widgets=[Container(widgets=[self.sample, self.SIM_mode, self.Polarisation, self.Acceleration, self.Psf, self.N,
                                        self.pixel_size, self.ill_NA, self.det_NA, self.n,
                                        self.ill_wavelength, self.det_wavelength]),
                     Container(widgets=[self.n_samples, self.magnification, self.zrange, self.tpoints, self.xdrift,
                                        self.zdrift, self.fwhmz, self.random_seed, self.drift, self.defocus,
                                        self.sph_abb])], layout="horizontal")
        w_cal = magicgui(self.get_results, call_button="Calculate raw image stack", auto_call=False)
        w_save_and_print = Container(widgets=[magicgui(self.save_tiff_with_tags, call_button='save_tiff_with_tags'),
                                              magicgui(self.print_tiff_tags, call_button='print_tags')],
                                     layout="horizontal", labels=None)
        w_sum = magicgui(self.show_raw_img_sum, auto_call=True)
        w_psf = magicgui(self.show_psf, auto_call=True)
        w_otf = magicgui(self.show_otf, auto_call=True)
        self.messageBox = LineEdit(value="Messages")
        self.w = Container(widgets=[w_parameters, w_cal, w_save_and_print, w_sum, w_psf, w_otf, self.messageBox],
                           labels=None)
