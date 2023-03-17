from napari_generic_simulator import SIMulator

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams


def test_simulator_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    simulator_widget = SIMulator(viewer)