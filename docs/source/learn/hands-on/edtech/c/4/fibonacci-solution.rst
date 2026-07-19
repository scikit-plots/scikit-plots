******************************
Lab 4: Fibonacci C Solution
******************************

.. include:: ../c-urls.rst

.. _fibonacci_c_solution:

Run this code on |ide.judge0.com|!

.. code-block:: c

    /*
     *  Name:
     *  Date:
     *  Project name:
     *
     *  Description: Describe what this file does
     */

    #include <stdio.h>
    #include <stdlib.h>

    #define MAX_FIB 47

    // Students: Give a brief description of the prototype and describe each parameter

    /**
      * Calculates the next Fibonacci number
      * unsigned int next:        The next number in the Fibonacci sequence
      * unsigned int *previous:   The pointer to the previous number in the Fibonacci sequence
      * return value:             The next number in the Fibonacci sequence (next + previous)
      */
    unsigned int calc_fibonacci(unsigned int, unsigned int *);

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
            printf("Usage: fibonacci [integer 0-%d]", MAX_FIB);
            return 0;
        }

        int count = atoi(argv[1]);

        if (count < 0 || count > MAX_FIB)
        {
            printf("Usage: fibonacci [integer 0-%d]", MAX_FIB);
            return 0;
        }

        unsigned int previous = 0;
        unsigned int next = 1;

        printf("%u \n", previous);

        for (int i = 0; i < count; i++)
        {
            printf("%u \n", next);

            next = calc_fibonacci(next, &previous);
        }

        return 0;
    }

    /*
     * Calculates the next Fibonacci number
     * int next:        The next number in the Fibonacci sequence
     * int *previous:   The pointer to the previous number in the Fibonacci sequence
     * return value:    The next number in the Fibonacci sequence (next + previous)
     */
    unsigned int calc_fibonacci(unsigned int next, unsigned int *previous)
    {
        unsigned int swap = next;
        next = *previous + next;
        *previous = swap;

        return next;
    }

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/c/4/fibonacci-solution.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
