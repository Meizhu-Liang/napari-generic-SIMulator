"""
@authors: Meizhu Liang @Imperial College
"""
from magicgui import magicgui
from magicgui.widgets import Container
from enum import Enum
from napari_generic_simulator.baseSIMulator import import_cp, import_torch, torch_GPU
from napari_generic_simulator.hexSIMulator import HexSim_simulator, RightHexSim_simulator
from napari_generic_simulator.conSIMulator import ConSim_simulator
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLineEdit
from napari.qt.threading import thread_worker

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
    if import_cp:
        USE_CUPY = 1
    if import_torch:
        USE_TORCH_CPU = 2
        if torch_GPU:
            USE_TORCH_GPU = 3

class SIMulator(QWidget):
    """
    A Napari plugin for the simulation of raw images produced while scanning an object (3D point cloud) through focus as
    the hexSIM illumination pattern is shifted through 7 positions laterally.The raw data is processed by a standard, a
    frame-by-frame and a batch reconstruction to produce the super-resolved output.https://doi.org/10.1098/rsta.2020.0162
    This currently supports hexagonal SIM (1 angle, 7 phases) with three beams and that at right-angles.
    """
    def __init__(self, viewer: 'napari.viewer.Viewer'):
        self._viewer = viewer
        super().__init__()
        self.parameters()
        self.setup_ui()
        self.start_simulator()

    def setup_ui(self):
        self.wrap_widgets()
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.add_magic_function(self.w, layout)
        self.messageBox = QLineEdit()
        layout.addWidget(self.messageBox, stretch=True)
        self.messageBox.setText('Messages')

    def add_magic_function(self, function, _layout):
        self._viewer.layers.events.inserted.connect(function.reset_choices)
        self._viewer.layers.events.removed.connect(function.reset_choices)
        _layout.addWidget(function.native)

    def parameters(self, SIM_mode=Sim_mode.HEXSIM_RIGHT_ANGLES, Polarisation=Pol.AXIAL, Acceleration=list(Accel)[-1],
                   N: int = 512, pixel_size_μm: float = 5.5, magnification: int = 60, NA: float = 1.1, n: float = 1.33,
                   wavelength_μm: float = 0.52, npoints: int = 500, zrange_μm: float = 7.0, dz_μm: float = 0.35,
                   fwhmz_μm: float = 3.0, random_seed: int = 123):
        self.SIM_mode = SIM_mode.value
        self.Polarisation = Polarisation.value
        self.Acceleration = Acceleration.value
        self.N = N
        self.pixel_size = pixel_size_μm
        self.magnification = magnification
        self.NA = NA
        self.n = n
        self.wavelength = wavelength_μm
        self.npoints = npoints
        self.zrange = zrange_μm
        self.dz = dz_μm
        self.fwhmz = fwhmz_μm
        self.random_seed = random_seed
        self.par_list =[self.SIM_mode, self.Polarisation, self.Acceleration, self.N, self.pixel_size, self.magnification, self.NA, self.n, self.wavelength,
                self.npoints, self.zrange, self.dz, self.fwhmz, self.random_seed]

    def set_att(self):
        """Sets attributes in the simulation class. Executed frequently to update the parameters"""
        if self.SIM_mode == Sim_mode.HEXSIM.value:
            self.sim = HexSim_simulator()
        elif self.SIM_mode ==Sim_mode.HEXSIM_RIGHT_ANGLES.value:
            self.sim = RightHexSim_simulator()
        elif self.SIM_mode == Sim_mode.SIM_CONV.value:
            self.sim = ConSim_simulator()

        if self.Polarisation == Pol.IN_PLANE.value:
            self.sim.pol = 0
        elif self.Polarisation == Pol.AXIAL.value:
            self.sim.pol = 1
        elif self.Polarisation == Pol.CIRCULAR.value:
            self.sim.circular = 2

        if self.Acceleration == Accel.USE_NUMPY.value:
            self.sim.acc = 0
        if hasattr(Accel, 'USE_CUPY'):
            if self.Acceleration == Accel.USE_CUPY.value:
                self.sim.acc = 1
        if hasattr(Accel, 'USE_TORCH_CPU'):
            if self.Acceleration == Accel.USE_TORCH_CPU.value:
                self.sim.acc = 2
        if hasattr(Accel, 'USE_TORCH_GPU'):
            if self.Acceleration == Accel.USE_TORCH_GPU.value:
                self.sim.acc = 3

        self.sim.N = self.N
        self.sim.pixel_size = self.pixel_size
        self.sim.magnification = self.magnification
        self.sim.NA = self.NA
        self.sim.n = self.n
        self.sim.wavelength = self.wavelength
        self.sim.npoints = self.npoints
        self.sim.zrange = self.zrange
        self.sim.dz = self.dz
        self.sim.fwhmz = self.fwhmz
        self.sim.random_seed = self.random_seed

    def start_simulator(self):
        """
        Starts the raw images generators and create the frequency space.

        """
        if hasattr(self, 'sim'):
            self.stop_simulator()
            self.start_simulator()
        else:
            self.set_att()

    def setReconstructor(self):
        pass

    def stop_simulator(self):
        if hasattr(self, 'sim'):
            delattr(self, 'sim')

    def show_img(self):
        self._viewer.add_image(data=self.sim.img, name='raw image stack')

    @thread_worker(connect={"returned": show_img})
    def get_results(self):
        self.set_att()
        t = self.sim.raw_image_stack()
        try:
            while True:
                self.messageBox.setText(next(t))
        except Exception as e:
            print(e)
        self.used_par_list = [self.SIM_mode, self.Polarisation, self.Acceleration, self.N, self.pixel_size,
                              self.magnification, self.NA, self.n, self.wavelength, self.npoints, self.zrange, self.dz,
                              self.fwhmz, self.random_seed]
        return self

    def show_raw_img_sum(self, show_raw_img_sum: bool=False):
        if show_raw_img_sum:
            if hasattr(self.sim, 'img_sum_z'):
                if self.used_par_list != self.par_list:
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
                if self.used_par_list != self.par_list:
                    self.messageBox.setText('Parameters changed!')
                else:
                    try:
                        self._viewer.add_image(self.sim.psf_z0, name='PSF in x-y plane')
                    except Exception as e:
                        print(e)

    def show_otf(self, show_3D_otf: bool=False):
        if show_3D_otf:
            if hasattr(self.sim, 'aotf_x'):
                if self.used_par_list != self.par_list:
                    self.messageBox.setText('Parameters changed! Calculate the raw-image stack first!')
                else:
                    try:
                        self._viewer.add_image(self.sim.aotf_x, name='OTF perpendicular to x')
                        self._viewer.add_image(self.sim.aotf_y, name='OTF perpendicular to y')
                    except Exception as e:
                        print(str(e))

    def wrap_widgets(self):
        w1 = magicgui(self.parameters, layout="vertical", auto_call=True)
        w2 = magicgui(self.get_results, call_button="Calculate raw image stack", auto_call=False)
        w3 = magicgui(self.show_raw_img_sum, auto_call=True)
        w4 = magicgui(self.show_psf, layout="vertical", auto_call=True)
        w5 = magicgui(self.show_otf, layout="vertical", auto_call=True)
        self.w = Container(widgets=[w1,w2, w3, w4, w5], labels=None)

if __name__ == '__main__':
    import napari
    viewer = napari.Viewer()
    test = SIMulator(viewer)
    viewer.window.add_dock_widget(test, name='my second app', add_vertical_stretch=True)

    napari.run()