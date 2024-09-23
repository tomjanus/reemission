Using the ReEmission Docker Image
=================================

.. _GeoCARET: https://github.com/Reservoir-Research/geocaret

The ReEmission Docker image is the simplest way to run ReEmission CLI and run ReEmission together with GeoCARET_. This guide will take you through the required steps:

1. Install Docker Desktop / Docker Engine (Linux only).
2. Pull the ReEmission Docker image.
3. Use Docker compose to run ReEmission.

   .. note::
      This documentation assumes that the reader: (1) Understands how to use ReEmission CLI. See :doc:`../usage`. (2) Has a basic familiarity with the shell (macOS or linux) or PowerShell (Windows).

Install Docker Desktop
----------------------

If you already have Docker installed on your computer then skip to the next section, `Pull the ReEmission Docker image`_.

Installing Docker Desktop
~~~~~~~~~~~~~~~~~~~~~~~~~

Docker Desktop is free and can be installed on Windows, Mac & Linux computers, Please visit https://docs.docker.com/get-docker/ and follow the appropriate instructions for installing Docker Desktop on your computer. For Linux distributions, you can alternatively install `Docker Engine <https://docs.docker.com/engine/install/>`_. 
Once installed, make sure Docker is running, and open a shell prompt (Linux/macOS) or PowerShell (Windows) and typing the following:

.. code-block:: bash

   docker -v

This should return the version number of the installed version of docker. If you see an error message along the lines of ‘Cannot connect to the Docker daemon’ then restart Docker Desktop and try again.

.. _Pulling ReEmission Docker image:

Pull the ReEmission Docker image
--------------------------------

Open a shell prompt (macOS/Linux) or PowerShell (Windows) and type:

.. code-block:: bash

   docker pull ghcr.io/tomjanus/reemission

Use Docker compose to run ReEmission
------------------------------------

Prepare your workspace
~~~~~~~~~~~~~~~~~~~~~~

Docker compose is a tool for simplifying the execution of docker containers. We’ll use it to run ReEmission.

First you’ll need to create a new (root) folder for your ReEmission workspace where you must create two sub-folders:

-  **examples**, which will hold your input data files and examples.
-  **outputs**, which will hold the analyses output files

You can create this folder structure in a shell prompt (Linux or macOS), or PowerShell if on Windows, by typing:

.. code-block:: bash

   mkdir my_reemission_work_folder
   cd my_reemission_work_folder
   mkdir examples
   mkdir outputs

You will also need to download the file `compose.yaml <https://github.com/tomjanus/reemission/blob/release/compose.yaml>`__ and save it inside your ReEmission workspace folder (e.g. ``my_reemission_work_folder`` in the above example). The final file structure (assuming your top folder is called *my_reemission_work_folder*)should look as follows:

::

    my_reemission_work_folder
    ├── examples
    ├── outputs
    └── compose.yml

.. important:: 

   **Linux users & directory permissions**

   When run on a linux host computer, the ReEmission docker image will only work if the user ID & group ID (``UID:GID``) of your user account is ``1000:1000``. Otherwise, ReEmission will not be able to write to the ``outputs/`` folder.
   If you use Linux on a personal laptop, then it is very likely your user account ``UID:GID`` will be ``1000:1000``. However, this may not be the case if you log in to a Linux server with multiple users. To check your user account, type:

   .. code-block:: bash

      id -u  # print user ID (UID)
      id -g  # print group ID (GID)

   If your user account has a different UID and/or GID then you should either change the UID or GID, respectively so that they're ``1000:1000``.

Test that ReEmission works
~~~~~~~~~~~~~~~~~~~~~~~~~~

To test everything is working correctly, you should first run the following from inside the ReEmission workspace folder you just created:

.. code-block:: bash

   cd my_reemission_work_folder
   docker compose run --rm reemission

You should see the output of the command line interface with usage instructions.

Running ReEmission with ``docker compose``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the ReEmission Docker container, please read the instructions in :doc:`../running_docker`.
