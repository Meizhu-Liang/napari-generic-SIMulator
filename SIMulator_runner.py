#!c:/Users/maan/Documents/HexSimProcessor/venv/Scripts/python.exe

from src.napari_generic_simulator._widget import SIMulator, PointCloud

if __name__ == '__main__':
    import napari

    viewer = napari.Viewer()

    point_cloud = PointCloud(viewer)
    simulator = SIMulator(viewer)
    w1 = viewer.window.add_dock_widget(point_cloud, name='Point cloud generator', add_vertical_stretch=True)
    w2 = viewer.window.add_dock_widget(simulator, name='SIM data generator', add_vertical_stretch=True)
    viewer.window._qt_window.tabifyDockWidget(w1, w2)

    napari.run()
