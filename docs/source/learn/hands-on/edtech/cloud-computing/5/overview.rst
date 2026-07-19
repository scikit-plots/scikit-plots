*****************
Lab 5: Overview
*****************

.. include:: urls.rst

.. contents:: Table of Contents

You've learned how to download a project from GitHub and then modify the
``Dockerfile`` to create a custom image. Now, we will explore how to create
our own Dockerfile to build a deployable Docker image.

**Writing a Dockerfile**

We will write our own ``Dockerfile`` to build an image to host a Python
application using the |Flask| microframework. This method allows us to bundle
an application in a state that includes the required packages and
configurations. It makes it easy to deploy the application!

Goals for Lab 5
==================
We will explore how to containerize applications so that we can run a service
using multiple smaller containers.

During this lab, you will learn how to:

 #. **create** a basic ``Dockerfile``
 #. **daemonize** a container to run in the background
 #. **package** a web application or internet service in a Docker image

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/cloud-computing/5/overview.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
