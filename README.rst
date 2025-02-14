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

If you are interested in collaborating with us on DFXM data analysis, please reach out to us at: naxhe@dtu.dk
and we can discuss the possibilities.

Usecase (following the v1.0.0 release)
------------------------------------------------

.. code-block:: python

   import darling
   import matplotlib.pyplot as plt
   path_to_data, _, _ = darling.assets.mosaicity_scan()
   reader = darling.reader.MosaScan(path_to_data)
   dset = darling.DataSet(reader)
   dset.load_scan(scan_id="1.1")
   background = dset.estimate_background()
   dset.subtract(background)
   mean, covariance = dset.moments()
   fig, ax = dset.plot.mosaicity()
   fig.suptitle('Mosaicity Map - a type of colorcoding of the $\chi$ - $\phi$ scan', fontsize=22, y=0.8)
   plt.show()


.. image:: ../../docs/source/images/mosa.png
   :align: center


Documentation
------------------------------------------------
Darling hosts documentation at https://axelhenningsson.github.io/darling/


Release Notes v1.0.0
------------------------------------------------
Darling v1.0.0 is now available on the main branch. 

This release features a simplified API for reading h5 files in which
much of the needed information is automatically extracted.

Moreover, the fundamental representation of the scan motors have been updated
such that the darling.properties module now operates on 2D grids of coordinates
in which the floating point values are allowed to be non-uniformly spaced.

This allows for proceesing of data with motor drift, and maximises the precision
of extracted moments.

For use of the old darling v0.0.0 consider checkout from and older commit:

.. code-block:: bash

   git clone https://github.com/AxelHenningsson/darling.git
   git checkout afe52
   cd darling
   pip install -e .


Installation
------------------------------------------------
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



