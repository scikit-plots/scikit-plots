Lab 7: Shell Scripting
======================

.. include:: urls.rst

.. contents:: Table of Contents

Shell scripting is an integral part of working with cloud computing. Many
components of cloud computing are 100% automated. These services must work
with minimal or no human interaction. Everything that we've done here
can be done using a shell script.

In Linux, many processes happen in the background on a set schedule.
Linux uses ``cron`` jobs.

* Cron is a time-based job scheduler in Unix-like computer operating
  systems.
* Cron enables users to schedule jobs (commands or shell scripts)
  that run periodically at certain times or dates.
* Cron job can run every minute, hour, day, week, month, or year
  or in any combination.

We are changing our focus from cloud-based services to working with Linux.
This lab requires working with the operating system to support the tools
instead of setting up a network or web service.

  * It is **strongly recommended** that you view the |Linux Tutorial|
    to give you an understanding of how Linux operates and some common
    commands.

**Additional Resources**

* |Linux Tutorial|
* |Shell Scripting Tutorial|
* |Ryan's Bash Scripting Tutorial|
* |A Beginners Guide To Cron Jobs|


Goals for Lab 7
---------------
During this lab, you will learn how to:

 #. **write** basic shells scripts.
 #. **create** an backup script for your Docker content.
 #. **automate** the script to create backups on a schedule.

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/cloud-computing/7-scripting/overview.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
