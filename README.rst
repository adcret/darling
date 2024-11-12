DARLING
------------------------------------
the python **[D]ark** field x-ray microscopy **[A] alysis** & **[R]econstruction** **[L]ibrary** for rapid data **[IN]spection** & **[G]raphing**

Darling is written and maintained by: 

Axel Henningsson,
Felix Tristan Frankus and 
Adam André William Cretton, 

affiliated with DTU. The core ideas of this library was originally written during a beamtime at ESRF id03D. If you use this code in your research we ask that you cite this repository.

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

.. image:: https://github.com/AxelHenningsson/darling/blob/main/docs/source/images/mosa.png?raw=true
   :align: center

Documentation
------------------------------------
Darling hosts documentation at https://axelhenningsson.github.io/darling/


Installation
------------------------------------
From source

.. code-block:: bash

    git clone https://github.com/AxelHenningsson/darling.git
    cd darling
    pip install -e .
