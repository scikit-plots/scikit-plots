.. _c-templates:

************
C Templates
************

These templates will help you start your projects.

.. _basic-c-template:

Basic C Template
=================

.. code-block:: c

    #include <stdio.h>

    int main()
    {
        printf("Hello, world!\n");

        return 0;
    }

Command Line Args C Template
=============================

.. code-block:: c

    #include <stdio.h>

    int main(int argc, char *argv[])
    {

        /**
        * - argc is the number of args
        *   - 0,2...n
        *   - 1 is invalid because argv[0] always contains the filepath
        * - argv[0] is the name a path of the executable
        * - argv[1] is first argument passed to the application
        * - argv[n] are the consecutive args
        */

        // Verify at least one arg
        if (argc < 2)
        {
            printf("Usage: program arg1");
            return 0;
        }

        // Display the program name and first arg
        printf("arg[0] --> %s\n", argv[0]);
        printf("arg[1] --> %s\n", argv[1]);

        return 0;
    }


.. _complete-c-template:

Complete C Template
=====================

.. code-block:: c

    /*
     *  Name:
     *  Date:
     *  Project name:
     *
     *  Description: Describe what this file does
     */

    #include <stdio.h>

    // Students: Give a brief description of the prototype and describe each parameter

    /*
     * Sums the values in an array
     * int array_size:     The size of the int array
     * int *array_pointer: The pointer to the int array
     * return value:       The summation of the values in the array
     */
    int sum_values_in_array(int, int *);

    // Students: Describe the number of args, and each required and optional arg in main()

    /*
    * Function: main
    * ----------------------------
    *   Application entry point
    *
    *   argc: Expected number of args: 1
    *         Optional args: 1
    *   argv[0]: 'a': Performs the action using an average
    *   argv[1]: 'b': performs the action using a summation
    */
    int main(int argc, char *argv[])
    {
        // Verify at least one arg
        if (argc < 2)
        {
            printf("Usage: program arg1");
            return 0;
        }

        // Display the program name and first arg
        printf("arg[0] --> %s\n", argv[0]);
        printf("arg[1] --> %s\n", argv[1]);

        return 0;
    }

    /*
    * Function: sum_values_in_array
    * -------------------------------
    *   Sums integer values in an array and then returns the result
    *
    *   size: The size of the int array
    *   int *array_pointer: The pointer to the int array
    *
    *   returns:
    *     - The summation of all values in the array.
    *     - 0 for an empty array, an error, or for a summation that results in 0
    */
    int sum_values_in_array(int size, int *array_pointer)
    {
        return 0;
    }

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/c/templates/index.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
