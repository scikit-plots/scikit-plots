Lab 8: VPS Security
===================

.. include:: urls.rst

.. contents:: Table of Contents

Any internet-based device can be hacked. A goal is to make it hard
for someone to gain access to your VPS. You have already taken the
first steps by configuring a firewall (UFW) and enabling Fail2Ban.

There are several other methods that you can use to add security to
your VPS with little effort. This lab gives you an overview of
these methods.

Here is a shortlist of security methods that you can implement quickly
to provide a wide range of protection.

* Configure a firewall
* Use Fail2Ban
* Harden SSH logins

  * Use a non-default port
  * Login using a non-privileged user
  * Restrict root access without a private key
* Perform automatic updates

**Security Resources**

* |Securing a VPS| from OVH
* |How to Harden your Ubuntu 18.04 Server| from HostAdvice
* |How to Secure Your Server| from Linode
* |5 Steps to Harden SSH Server on Ubuntu| from LinuxBabe
* |National Cyber Security Centre publish Ubuntu 18.04 LTS Security Guide|


Goals for Lab 8
---------------
During this lab, you will learn how to:

 #. **enable** automatic updates.
 #. **harden** SSH logins.

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/cloud-computing/8/overview.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
