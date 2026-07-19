*********************
Miscellaneous Code
*********************

Creating files with Test Code
=============================
Sometimes we need to create files with text to help us develop or test
our code.

Bash or CMD
-------------
One ways to create a file in the command line is to use  ``echo``

   .. code-block:: bash

      echo "some text" > filename.txt

system() function
--------------------
.. caution:: This method is not intended for production code

#. You can wrap the command in function ``system()`` to run a command
   from your code.

   .. code-block:: c

       system("echo \"some text\" > filename.txt");

#. You can also use ``sprintf`` to build you command string using variables.
   ``sprintf`` works similarly to ``printf`` except it sends
   the data to the a variable instead to the screen.

   .. code-block:: c

       char *filename = "filename.txt";
       char command[128];                   // container for the command

       // Build a string and store in variable 'command'
       sprintf(command, "echo \"%s\" > %s", "some random text", filename);

       // Execute the command
       system(command);

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/operating-systems/references/misc.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
