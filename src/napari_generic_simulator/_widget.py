"""
Widget of the napari plugin
"""
__author__ = "Meizhu Liang @Imperial College London"

from magicgui import magicgui
from magicgui.widgets import Container
from enum import Enum
from napari_generic_simulator.baseSIMulator import import_cp, import_torch, torch_GPU
from napari_generic_simulator.hexSIMulator import HexSim_simulator, RightHexSim_simulator
from napari_generic_simulator.conSIMulator import ConSim_simulator
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLineEdit
from napari.qt.threading import thread_worker
from magicgui.widgets import SpinBox, FileEdit, Slider, Label, Container, ComboBox, FloatSpinBox

class Sim_mode(Enum):
    HEXSIM = 0
    HEXSIM_RIGHT_ANGLES = 1
    SIM_CONV = 2  # Conventional 2-beam SIM

class Pol(Enum):
    IN_PLANE = 0
    AXIAL = 1
    CIRCULAR = 2

class Accel(Enum):
    USE_NUMPY = 0
    if import_torch:
        USE_TORCH_CPU = 1
        if torch_GPU:
            USE_TORCH_GPU = 2
    if import_cp:
        USE_CUPY = 3




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
        self.messageBox = QLineEdit()
        layout.addWidget(self.messageBox, stretch=True)
        self.messageBox.setText('Messages')

    def add_magic_function(self, function, _layout):
        """Adds the widget to the viewer"""
        self._viewer.layers.events.inserted.connect(function.reset_choices)
        self._viewer.layers.events.removed.connect(function.reset_choices)
        _layout.addWidget(function.native)
    #
    # @magicgui(
    #     # call_button="Calculate",
    #     layout="vertical",
    #     # result_widget=True,
    #     # numbers default to spinbox widgets, but we can make
    #     # them sliders using the `widget_type` option
    #     n_points={"widget_type": "SpinBox", "min":0, "max":10000},
    #     # slider_float={"widget_type": "FloatSlider", "max": 100},
    #     # slider_int={"widget_type": "Slider", "readout": False},
    #     # radio_option={
    #     #     "widget_type": "RadioButtons",
    #     #     "orientation": "horizontal",
    #     #     "choices": [("first option", 1), ("second option", 2)],
    #     # },
    #     # filename={"label": "Pick a file:"},  # custom label
    # )


    def parameters(self):
        self.npoints = SpinBox(value=10, name='spin', label='Value:', min=-100, max=10000)
        self.SIM_mode = ComboBox(value=Sim_mode.HEXSIM_RIGHT_ANGLES, label='SIM_mode', choices=Sim_mode)
        self.Polarisation = ComboBox(value=Pol.AXIAL, label='Polarisation', choices=Pol)
        self.Acceleration = ComboBox(value=list(Accel)[-1], label='Acceleration', choices=Accel)
        self.N = SpinBox(value=128, name='spin', label='N pixel')
        self.pixel_size = FloatSpinBox(value=6.5, name='spin', label='pixel size(μm)', step=0.5)
        self.magnification = SpinBox(value=60, name='spin', label='objective magnification')
        self.NA = FloatSpinBox(value=1.1, name='spin', label='NA', min=0.0)
        self.n = FloatSpinBox(value=1.33, name='spin', label='n', min=0.00)
        self.wavelength = FloatSpinBox(value=0.60, name='spin', label='wavelength(μm)', min=0.00)
        self.n_points = SpinBox(value=500, name='spin', label='N points', max=10000, step=10)
        self.zrange = FloatSpinBox(value=3.5, name='spin', label='z range(μm)', min=0.0)
        self.dz = FloatSpinBox(value=0.35, name='spin', label='dz(μm)', min=0.00)
        self.fwhmz = FloatSpinBox(value=3.0, name='spin', label='fwhmz(μm)', min=0.0)
        self.random_seed = SpinBox(value=123, name='spin', label='random seed')
        self.drift = FloatSpinBox(value=0.0, name='spin', label='drift(μm)', min=-10.0)
        self.defocus = FloatSpinBox(value=0.0, name='spin', label='de-focus aberration(μm)', min=-10.0)
        self.sph_abb = FloatSpinBox(value=0.0, name='spin', label='spherical aberration', min=-10.0)

    def par_list(self):
        """return the current parameter list"""
        return [self.SIM_mode.value, self.Polarisation.value, self.Acceleration.value, self.N.value,
                         self.pixel_size.value, self.magnification.value, self.NA.value, self.n.value,
                         self.wavelength.value, self.n_points.value, self.zrange.value, self.dz.value,
                         self.fwhmz.value, self.random_seed.value, self.drift.value, self.defocus.value,
                         self.sph_abb.value]
    def set_att(self):
        """Sets attributes in the simulation class. Executed frequently to update the parameters"""
        if self.SIM_mode.value == Sim_mode.HEXSIM:
            self.sim = HexSim_simulator()
        elif self.SIM_mode.value ==Sim_mode.HEXSIM_RIGHT_ANGLES:
            self.sim = RightHexSim_simulator()
        elif self.SIM_mode.value == Sim_mode.SIM_CONV:
            self.sim = ConSim_simulator()

        if self.Polarisation.value == Pol.IN_PLANE:
            self.sim.pol = 'in-plane'
        elif self.Polarisation.value == Pol.AXIAL:
            self.sim.pol = 'axial'
        elif self.Polarisation.value == Pol.CIRCULAR:
            self.sim.pol = 'circular'

        if self.Acceleration.value == Accel.USE_NUMPY:
            self.sim.acc = 0

        if hasattr(Accel, 'USE_TORCH_CPU'):
            if self.Acceleration.value == Accel.USE_TORCH_CPU:
                self.sim.acc = 1
        if hasattr(Accel, 'USE_TORCH_GPU'):
            if self.Acceleration.value == Accel.USE_TORCH_GPU:
                self.sim.acc = 2
        if hasattr(Accel, 'USE_CUPY'):
            if self.Acceleration.value == Accel.USE_CUPY:
                self.sim.acc = 3

        self.sim.N = self.N.value
        self.sim.pixel_size = self.pixel_size.value
        self.sim.magnification = self.magnification.value
        self.sim.NA = self.NA.value
        self.sim.n = self.n.value
        self.sim.wavelength = self.wavelength.value
        self.sim._ = self.n_points.value
        self.sim.zrange = self.zrange.value
        self.sim.dz = self.dz.value
        self.sim.fwhmz = self.fwhmz.value
        self.sim.drift = self.drift.value
        self.sim.random_seed = self.random_seed.value
        self.sim.defocus = self.defocus.value
        self.sim.sph_abb = self.sph_abb.value
        self.used_par_list = [self.SIM_mode.value, self.Polarisation.value, self.Acceleration.value, self.N.value,
                              self.pixel_size.value, self.magnification.value, self.NA.value, self.n.value,
                              self.wavelength.value, self.n_points.value, self.zrange.value, self.dz.value,
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

    def show_img(self):
        self._viewer.add_image(data=self.sim.img, name='raw image stack')

    @thread_worker(connect={"returned": show_img})
    def get_results(self):
        self.set_att()
        if self.sim.drift != 0.0:
            t = self.sim.raw_image_stack_brownian()
        else:
            t = self.sim.raw_image_stack()
        try:
            while True:
                self.messageBox.setText(next(t))
        except Exception as e:
            print(e)
        return self

    def show_raw_img_sum(self, show_raw_img_sum: bool=False):
        if show_raw_img_sum:
            if hasattr(self.sim, 'img_sum_z'):
                if self.used_par_list != self.par_list():
                    self.messageBox.setText('Parameters changed! Calculate the raw-image stack first!')
                else:
                    try:
                        self._viewer.add_image(self.sim.img_sum_z, name='raw image sum along z axis')
                        self._viewer.add_image(self.sim.img_sum_x, name='raw image sum along x (or y) axis')
                    except Exception as e:
                        print(str(e))

    def show_psf(self, show_3D_psf: bool=False):
        if show_3D_psf:
            if hasattr(self.sim, 'psf_z0'):
                if self.used_par_list != self.par_list():
                    self.messageBox.setText('Parameters changed! Calculate the raw-image stack first!')
                else:
                    try:
                        self._viewer.add_image(self.sim.psf_z0, name='PSF in x-y plane')
                    except Exception as e:
                        print(e)

    def show_otf(self, show_3D_otf: bool=False):
        if show_3D_otf:
            if hasattr(self.sim, 'aotf_x'):
                if self.used_par_list != self.par_list():
                    self.messageBox.setText('Parameters changed! Calculate the raw-image stack first!')
                else:
                    try:
                        self._viewer.add_image(self.sim.aotf_x, name='OTF perpendicular to x')
                        self._viewer.add_image(self.sim.aotf_y, name='OTF perpendicular to y')
                    except Exception as e:
                        print(str(e))

    # def pr(self):
    #     print(self.npoints.value)

    def wrap_widgets(self):
        """Creates a widget containing all small widgets"""
        # w1 = magicgui(self.parameters, layout="vertical", auto_call=True)
        w1 = Container(widgets=[self.SIM_mode, self.Polarisation, self.Acceleration, self.N, self.pixel_size,
                              self.magnification, self.NA, self.n, self.wavelength, self.n_points, self.zrange, self.dz,
                              self.fwhmz, self.random_seed, self.drift, self.defocus, self.sph_abb])
        w2 = magicgui(self.get_results, call_button="Calculate raw image stack", auto_call=False)
        # w2 = magicgui(self.pr, call_button="Calculate raw image stack", auto_call=False)
        w3 = magicgui(self.show_raw_img_sum, auto_call=True)
        w4 = magicgui(self.show_psf, layout="vertical", auto_call=True)
        w5 = magicgui(self.show_otf, layout="vertical", auto_call=True)
        self.w = Container(widgets=[w1, w2, w3, w4, w5], labels=None)

if __name__ == '__main__':
    import napari
    viewer = napari.Viewer()
    test = SIMulator(viewer)
    viewer.window.add_dock_widget(test, name='my second app', add_vertical_stretch=True)

    napari.run()