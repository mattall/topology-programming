.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/topology-programming.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/topology-programming
    .. image:: https://readthedocs.org/projects/topology-programming/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://topology-programming.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/topology-programming/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/topology-programming
    .. image:: https://img.shields.io/pypi/v/topology-programming.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/topology-programming/
    .. image:: https://img.shields.io/conda/vn/conda-forge/topology-programming.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/topology-programming
    .. image:: https://pepy.tech/badge/topology-programming/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/topology-programming
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/topology-programming

.. .. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
..     :alt: Project generated with PyScaffold
..     :target: https://pyscaffold.org/

.. |

====================
topology-programming
====================

Install
=======

Install the package with `pip install .`
You may want to run this within a virtual environment. 

Note: **use pip version 22.3.1**, *do not use 23.1.2*.

If you have a later version, you can use the command bellow.
`python -m pip install --upgrade pip==22.3.1``

In later versions one dependency, tmgen will not install.

Run
=======

The main program is ``src/onset/simulator.py``

Examples that load and run the ``Simulation`` module are in ``scrips/``.

The simulator can be run from the command line with ``src/onset/net_sim.py``.

The command-line version of the program expects 3 arguments minimum.

1. A network name, this name should have a ``.gml`` or ``.json`` file containing the network graph in ``data/graphs/{json, gml}/`` (the file suffix is *not* expected by the command line interpreter).

2. The number of nodes in the network.

3. A name for the experiment to run. This will inform the program where to save the output and results within ``data/results/``.

There are more arguments that you may pass. Changing the set of parameters will also change the output file destination.
A full list of arguments and their description can be viewed by running ``python src/onset/net_sim.py --help``.


