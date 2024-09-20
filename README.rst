========
Overview
========

Python package for calculating GHG emissions from man-made reservoirs

* Free software: GPL-3.0 license

Installation
============

::

    pip install reemission

You can also install the in-development version with::

    pip install git+ssh://git@tomjanus/reemission/tomjanus/reemission.git@main

Documentation
=============


https://reemission.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
