from src.napari_generic_simulator._widget import SIMulator

if __name__ == '__main__':
    import napari
    viewer = napari.Viewer()
    test = SIMulator(viewer)
    viewer.window.add_dock_widget(test, name='my second app', add_vertical_stretch=True)

    napari.run()