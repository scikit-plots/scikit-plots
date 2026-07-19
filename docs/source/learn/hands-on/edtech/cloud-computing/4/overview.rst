******************
Lab 4: Overview
******************
.. contents:: Table of Contents

You've learned how to start a Docker container using ``docker run`` and
``docker-compose``. In each instance, we used an existing image from
|Docker Hub| to create the container. We had a problem with the
WordPress image because the image itself did not support SSL.
Now, we need to learn how to build our own images using ``docker build``.

**Docker Build**

We will build an image from a ``Dockerfile``. The file contains the
set of instructions or commands a developer would type into a shell
session to configure a server. However, the process is 100% automated.
The ``Dockerfile`` contains all of the information necessary to
configure the image so that it can start as a container in a
preconfigured state.

You will learn how to extend an existing image to include additional
functionality, such as adding SSL.

.. |Docker Hub| raw:: html

   <a href="https://hub.docker.com/" target="_blank">Docker Hub</a>


Goals for Lab 4
=================
This lab demonstrates how to build an image from an existing ``Dockerfile``.

During this lab, you will learn how to:

 #. **create** a ``Dockerfile``
 #. **build** a Docker image from a ``Dockerfile``
 #. **extend** an image to create a modified version of the image
 #. **configure** an image and container with SSL support

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/cloud-computing/4/overview.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
