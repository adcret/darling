DARLING
------------------------------------
the python **[D]ark** field x-ray microscopy **[A]nalysis** & **[R]econstruction** **[L]ibrary** for rapid data **[IN]spection** & **[G]raphing**

.. image:: https://img.shields.io/badge/platform-cross--platform-brightgreen.svg
   :target: https://www.python.org/
   :alt: cross-platform

.. image:: https://img.shields.io/badge/code-pure%20python-blue.svg
   :target: https://www.python.org/
   :alt: pure python

.. image:: https://github.com/AxelHenningsson/darling/actions/workflows/pytest-linux-py310.yml/badge.svg
   :target: https://github.com/AxelHenningsson/darling/actions/workflows/pytest-linux-py310.yml
   :alt: tests ubuntu-linux

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: code style black

.. image:: https://img.shields.io/badge/docs-sphinx-blue.svg
   :target: https://axelhenningsson.github.io/darling/
   :alt: Sphinx documentation

Authors
------------------------------------
``darling`` is written and maintained by: 

`Axel Henningsson <https://github.com/AxelHenningsson>`_,
`Felix Tristan Frankus <https://github.com/adcret>`_ and
`Adam André William Cretton <https://github.com/fetrifra>`_

affiliated with DTU. The core ideas of this library was originally written during a beamtime at ESRF id03D. 

Until an associated journal publication is available, if you use this code in your research, we ask that you cite this repository.

Usecase
------------------------------------

.. code-block:: python

    import darling
    path_to_data, _, _ = darling.assets.mosaicity_scan()
    reader = darling.reader.MosaScan(
        path_to_data,
        ["instrument/chi/value", "instrument/diffrz/data"],
        motor_precision=[3, 3])
    dset = darling.DataSet(reader)
    dset.load_scan("instrument/pco_ff/image", scan_id="1.1")
    background = dset.estimate_background()
    dset.subtract(background)
    mean, covariance = dset.moments()
    dset.plot.mosaicity()

.. image:: ./darling/blob/main/docs/source/images/mosa.png?raw=true
   :align: center

Documentation
------------------------------------
Darling hosts documentation at https://axelhenningsson.github.io/darling/


Installation
------------------------------------
From source the key is simply to clone and pip install

.. code-block:: bash

    git clone https://github.com/AxelHenningsson/darling.git
    cd darling
    pip install -e .

In general, you probably want to install in a fresh virtual environment as

.. code-block:: bash

   python3 -m venv .venv_darling
   source .venv_darling/bin/activate
   git clone https://github.com/AxelHenningsson/darling.git
   cd darling
   pip install -e .

use 

.. code-block:: bash

   source .venv_darling/bin/activate

whenever you want to activate the environment. To add your env into a jupyter kernel such that
you can use it in an interactive notebook you may add the following two commands:

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user --name=darling

The following snippet has been verified to work on the ESRF slurm cluster 19 Dec 2024 in a browser terminal:

.. code-block:: bash

   python3 -m venv .venv_darling
   source .venv_darling/bin/activate
   git clone https://github.com/AxelHenningsson/darling.git
   cd darling
   pip install -e .
   pip install ipykernel
   python -m ipykernel install --user --name=darling



