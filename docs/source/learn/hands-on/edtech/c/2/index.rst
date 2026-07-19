**********************
Lab 2: Prime Numbers
**********************

.. include:: ../../includes/prolog.inc

.. include:: ../c-urls.rst

.. contents:: Table of Contents

You will learn about **preprocessor directives** in C.

Read These Pages:

- https://www.javatpoint.com/c-preprocessor
- https://www.tutorialspoint.com/cprogramming/c_preprocessors.htm
- https://www.tutorialspoint.com/cprogramming/c_header_files.htm
- https://docs.microsoft.com/en-us/cpp/preprocessor/hash-if-hash-elif-hash-else-and-hash-endif-directives-c-cpp?view=vs-2019


.. note:: You will need to install a GCC compiler on your computer.

    - See :ref:`Windows 10 C Environment` on how to set up Windows 10
      to compile and execute C.
    - Don't know if you have GCC installed? Open a prompt and run command:

      .. code-block:: bash

          gcc --version

          # Ubuntu shell Output
          user@DESKTOP-Z4T37ER:~$ gcc --version
          gcc (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0
          Copyright (C) 2017 Free Software Foundation, Inc.

          # Windows CMD Output
          C:\Users\user>gcc --version
          gcc (MinGW.org GCC-8.2.0-3) 8.2.0
          Copyright (C) 2018 Free Software Foundation, Inc.

Overview
==========
You will create a C program that determines if a given number is a
prime number.

A |prime number| is a a whole number that cannot be made by multiplying
other whole numbers.
less than or equal to *n*:

Task 1: Create a bool.h file
==========================================
C does not have a native ``bool`` type, there is a library called
``stdbool.h`` that offers that functionality. Instead of using it, let's
build our own!

.. note:: After this exercise, the safest option is to use ``stdbool.h`` :)
    This activity provides an example of how to create a simple ``.h`` file
    that contains a custom data type.

Traditional programming treats a 0 as FALSE and 1 as TRUE.
However, any non-zero value in C is TRUE, and **only 0 is FALSE**.
That means that expressions like ``int a = -8`` and ``int = 2350``
are evaluated as TRUE.

#. Read the discussion on on |Using boolean value in C| on Stack Overflow.
#. Choose option of creating your ``bool`` type.
#. Create file called ``mybool.h``
#. Create a new data type, such as ``my_bool``

Task 2: Create your Prime Function
==========================================

#. Start with the command line template from :ref:`c-templates`.
#. Include your ``mybool.h`` file:

   .. code-block:: C

       #include "mybool.h"

   a. Verify that your file compiles without any errors or warnings.
   b. See |How to write your own header file in C| for additional information.

#. Pass the value in using the command line
#. Create a function with this prototype (or using your own type):

   .. code-block:: C

        my_bool number_is_prime(int);

#. Develop the function. You can use Google to help you.
#. Print the result in ``main()`` if the number is prime or not

Task 3: Test Random Generated Numbers
==========================================
#. Implement a random number algorithm using https://www.geeksforgeeks.org/rand-and-srand-in-ccpp/
#. Use the ``#define`` macro to set the maximum number of random
   numbers to find. https://www.programiz.com/c-programming/c-preprocessor-macros

   .. code-block:: C

       #define MAX   25

       int main() {
       . . .
       for (i = 0; i < MAX; ++i)
       {
           // get random number
       . . .

#. Generate 25 numbers in a loop, test each number for prime.
#. Were any numbers prime?
#. If not, increase the value till you find one!

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/c/2/index.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
