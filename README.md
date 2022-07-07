# napari-generic-SIMulator

[![License BSD-3](https://img.shields.io/pypi/l/napari-generic-SIMulator.svg?color=green)](https://github.com/Meizhu-Liang/napari-generic-SIMulator/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-generic-SIMulator.svg?color=green)](https://pypi.org/project/napari-generic-SIMulator)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-generic-SIMulator.svg?color=green)](https://python.org)
[![tests](https://github.com/Meizhu-Liang/napari-generic-SIMulator/workflows/tests/badge.svg)](https://github.com/Meizhu-Liang/napari-generic-SIMulator/actions)
[![codecov](https://codecov.io/gh/Meizhu-Liang/napari-generic-SIMulator/branch/main/graph/badge.svg)](https://codecov.io/gh/Meizhu-Liang/napari-generic-SIMulator)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-generic-SIMulator)](https://napari-hub.org/plugins/napari-generic-SIMulator)

A napari plugin to simulate raw-image stacks of Structured illumination microscopy (SIM) with light sheet. The simulation can be GPU-accelerated if cupy is installed.   
Currently applies to:
- conventional 2-beam SIM data with 3 angles and 3 phases
- 3-beam hexagonal SIM data with 7 phases 
- 3-beam hexagonal SIM data with 7 phases at right-angles
----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/index.html
-->

## Installation

You can install `napari-generic-SIMulator` via [pip]:

    pip install napari-generic-SIMulator



To install latest development version :

    pip install git+https://github.com/Meizhu-Liang/napari-generic-SIMulator.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-generic-SIMulator" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/Meizhu-Liang/napari-generic-SIMulator/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
