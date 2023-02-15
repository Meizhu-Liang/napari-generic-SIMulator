from src.napari_generic_simulator._widget import SIMulator, PointCloud


if __name__ == '__main__':
    import napari

    viewer = napari.Viewer()

    point_cloud = PointCloud(viewer)
    simulator = SIMulator(viewer)
    viewer.window.add_dock_widget(point_cloud, name='Point cloud generator', add_vertical_stretch=True)
    viewer.window.add_dock_widget(simulator, name='SIM data generator', add_vertical_stretch=True)

    napari.run()