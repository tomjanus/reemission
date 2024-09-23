Building the ReEmission Docker Image
====================================

.. _ReEmission: https://github.com/tomjanus/reemission
.. _git: https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F

Docker can be used to quickly & easily create a run-time environment for running ReEmission_, particularly via its command line interface (CLI) without having to install the correct version of **Python** and the required **Python packages**.

To use ReEmission with Docker you will need to build and run an instance of a Docker image - see :ref:`What are Docker Images`.

Generally, you will not need to build your own image, as we provide a **pre-built image** as a package on GitHub, that will be suitable for most purposes. See :ref:`Pulling ReEmission Docker image` for accessing & using the pre-built ReEmission docker image.

However, if you want to build your own image, this guide will take you through the process.

As a prerequisite, you need to first install Docker `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ - an application that provides an easy-to-use interface for working with Docker on a desktop operating system. Docker Desktop is free and can be installed on Windows, Mac & Linux computers, Please visit https://docs.docker.com/get-docker/ and follow the appropriate instructions for installing Docker Desktop on your computer. For Linux distributions, you can alternatively install `Docker Engine <https://docs.docker.com/engine/install/>`_. 

.. note::
   This documentation assumes that the reader is comfortable working in the shell (linux/macOS) or PowerShell (Windows) and has
   basic experience with git_.

Once installed, make sure that Docker is running. To do so, open a shell prompt (Linux/macOS) or PowerShell (Windows) and type the following:

.. code-block:: bash

   docker -v

This should return the version number of the installed version of docker. If you see an error message along the lines of *‘Cannot connect to the Docker daemon’* then restart Docker Desktop / Docker Engine and try again.

Building the ReEmission Docker image
------------------------------------

Clone the ReEmission GitHub repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First clone the ReEmission GutHub repository to your computer:

.. code-block:: bash

   git clone https://github.com/tomjanus/reemission
   
Alternatively, if you don't use git_, download the package from the `GitHub page <https://github.com/tomjanus/reemission>`_ and extract to the working folder.

.. _build-the-reemission-docker-image-1:

Build the ReEmission Docker Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building the ReEmission image is simple. Make sure that 

* Docker Desktop / Docker Engine is running
* You are in the root folder of the ReEmission code base that you cloned in the previous step. 

Then inside the root directory of ReEmission, i.e. where `setup.py` and `setup.cfg` are located, type:

.. code-block:: bash

   docker build -t reemission_image .

Building the image might take a short while. The resulting image will be stored in a dedicated Docker folder on your computer. If you open Docker Desktop and go to the **‘Images’** section, you should see the **‘reemission_image’** image in the list.

You can now refer to :doc:`using_image` for details on how to run ReEmission CLI using your newly built Docker image.
