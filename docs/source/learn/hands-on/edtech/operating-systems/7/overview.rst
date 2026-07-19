Lab 7: Overview
===============

.. include:: ../c-urls.rst

This lab guides you through the process of terminating a process in
Windows using the system call ``TerminateProcess()`` and then retrieving
the exit code using ``GetExitCodeProcess()``.

Lab 7 Resources
---------------

- |Terminating a Process|
- |OpenProcess| function
- |CloseHandle| function
- |TerminateProcess| function
- |GetExitCodeProcess| function
- |Process Security and Access Rights|

Goals for Lab 7
---------------
During this lab, you will learn how to use:

 #. ``OpenProcess()`` to get the handle of a running process.
 #. ``TerminateProcess()`` to terminate a process forcefully.
 #. ``GetExitCodeProcess()`` to retrieve the exit code of a process

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/operating-systems/7/overview.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
