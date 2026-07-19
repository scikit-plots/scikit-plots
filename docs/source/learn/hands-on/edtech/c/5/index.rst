**************************
Lab 5: Dynamic Arrays
**************************

.. include:: ../../includes/prolog.inc

.. include:: ../c-urls.rst

.. contents:: Table of Contents

Overview
==========
You will write a program that converts text from ASCII to Integers
and stores it in a dynamic array.

C Memory Management
--------------------------
.. note::
    C does not automatically allocate and release memory except for variables
    and arrays declared directly.

C knows how much memory to allocate for these instances because the size
is declared in the source code. You cannot change the size of an array
in C because C allocates specifies the amount of memory to be allocated
the memory at compile time, also called static memory allocation.

  .. code-block:: C

      // The size is known. The compile specifies the memory size to allocate
      int index = 0;            // allocates 2 or 4 bytes
      int array[5] = {'0'}      // allocates 10 or 20 bytes

      // The pointer is the starting address. The size is not known.
      // This code will crash the program or destroy data in memory
      // because the memory was not allocated. It will overwrite
      // whatever is there.
      char *input;              // allocates a single byte
      scanf("%s",&input);       // could cause a segmentation fault

Programmers must use |dynamic memory allocation| for dynamic data.
For example, you don't want to create an array for the maximum or
worst-case scenario. Instead, a C programmer will use ``malloc`` to
allocate the required memory during runtime. Pointers are necessary to
allocate memory dynamically, which is the starting address of the
memory block.

  .. code-block:: C

      int index = 0;
      int array[5] = {'0'}

C does not preserve the end of the block or the size. Programmers using
C must manage the memory carefully to prevent issues, such as
|segmentation fault|'s and |memory leak|'s.

.. tip::
    C programmers must master the use of pointers to program in C
    effectively. Take a look at the code for a |linked list|. Note the
    use of pointers.

C has four functions to help you manage the memory.

  - |C - Memory Management|- An overview
  - |C Functions - malloc(), calloc(), realloc(), and free()|
  - |C Reference function malloc()|
  - |C Reference function calloc()|
  - |C Reference function realloc()|
  - |C Reference function free()|

C Memory Access Using Pointers
-------------------------------
.. caution::
    It is easy to get a |segmentation fault| when using pointers
    to access data in memory. C has no protection to prevent memory
    access violations.

You must always allocate the memory you need or know specifically the
size or range of the data.

C Memory Cleanup
--------------------------
C has no automatic memory cleanup, de-allocation, or garbage
collector features. The programmer must be attentive to allocate memory
and then free the memory when it is no longer needed. Otherwise,
your program will suffer from a |memory leak|. A memory leak is when an
application does not free allocated memory, which then uses more and more
memory the longer that the program runs.

Here is an example of a memory leak in C. The function allocates
memory using ``malloc`` but never frees that memory.

    .. code-block:: C
        :caption: Source: |Memory leak| (Wikipedia)

        void function_which_allocates(void)
        {
            /* allocate an array of 45 floats */
            float *a = malloc(sizeof(float) * 45);

            /* additional code making use of 'a' */

            /* return to main, having forgotten to free the memory we malloc'd */
        }

        int main(void) {
            function_which_allocates();

            /* the pointer 'a' no longer exists, and therefore cannot be freed,
            but the memory is still allocated. a leak has occurred. */
        }

Task 1: Allocate Memory for your Name
==========================================
First, let's prevents a a |segmentation fault| by allocation memory for
the name that the user enters. Otherwise, a user could in a long name
that the process tries to write to memory that it hasn't allocated

   .. warning::
       You cannot safely use a pointer array with ``scanf`` because
       it can cause an access memory violation error on strings
       longer than 12 or 16. The OS does know how much memory
       to allocate the program.


#. Open up |ide.judge0.com| or VSC and GCC. Start with a
   template, such as the :ref:`basic-c-template`.

#. Prompt the user for their name or text.

   .. caution::
       Creating a pointer for an array  without allocating will
       cause a |segmentation fault|.

       .. code-block:: c

           // a pointer to a single byte
           char *text;
           // Could cause segmentation fault if the input is larger than a byte
           scanf(&text);

   a. ``scanf`` stops reading input on the first whitespace. This site
      explains how to use ``scanf()`` and ``fgets()`` to
      |read in text that contains whitespace|.
   #. Allocate 64 bytes for the text by creating an empty array.
   #. Limit the length of the user input to 64 characters.

      .. code-block:: C

          // 1. Create an array using [] if using scanf
          //    arg[1] already allocates the required memory
          // 2. Use scanf or fgets to read the user input

          scanf("%s64", &text);
          fgets(text, MAX_LENGTH, stdin);

#. Print the text to verify that you are working with the correct data.

   .. code-block:: text

       Enter your name: Jimmy
       Jimmy


Task 2: Get the length of your name
==========================================
Next, let's get the length of the entered name.

Unfortunately, there is no way to calculate the length of an array from
a pointer. C only knows the start address. You can read more about the
problem here on StackOverflow post
|How to find the 'sizeof' (a pointer pointing to an array)|.

You can find the length of ``char *array = "abc"`` because a string always
terminates with the null pointer (``\0``). "abc" is the same as
``{'a','b','c','\0',}``. The solution is iterate through char array
moving the pointer to the next index looking for char ``\0``. This method
is probably how function ``strlen()`` finds the length of the string.

.. note:: Do not use ``strlen()``.

#. Create a function using this prototype:

   .. code-block:: C

       /**
       * Gets the size of a char array
       *
       * char *array: pointer to a char array
       * returns: the length of the char array
       */
       int get_size (char *);

#. Iterate through the char array **to count the number of characters**.
   The function will have this algorithm:

   .. code-block:: C

       /**
       * 1. Create a new pointer for the loop using the address to the array
       *    - You will lose the start address of *array if you advance that pointer
       *    char *ptr = array;
       *
       * 2. Loop through memory counting the characters until the contents
       *    of memory = '\0'
       *    - ptr is the address of the first character of the array
       *    - *ptr contains the data at that pointer location
       *    - ptr++ increments the pointer to the next address in memory
       *    - Compare the data at the memory address (*ptr) to '\0'
       *
       * 3. Return the length of of array
       */

       for (ptr; FILL_IN; ptr++) {

       }

#. Print the length of the string to verify that it works with strings
   of different lengths

   .. code-block:: text

       Enter your name: Jimmy
       Jimmy: 5

       Enter your name: Jimmy-the-Memory-Leaker
       Jimmy-the-Memory-Leaker: 23

Task 3: Convert your name to Integers
==========================================
You will dynamically allocate memory for an array that will hold
the Integer characters of your name using ``malloc``.

  - |C - Memory Management| (tutorialspoint)
  - |C Reference function malloc()|

#. Create a function with this prototype:

   .. code-block:: C

       /**
        * Creates an int array from a char array
        *
        * char *str_array: pointer to a char array
        * returns: the pointer to the int array
        */
       int *create_int_array(char *);

#. Iterate through the char array by advancing the pointer.

   a. You can advance the pointer in a ``for`` loop like this:

      .. code-block:: C

          // Add 'i' to the pointer and then get the value at that address
          int value = *(ptr + i);

          // prt++;       moves the pointer
          // *(ptr + i)   uese the offset to obtain the value

   #. Print the int values to the screen to verify that you have the correct
      values.

      .. code-block:: text
          :caption: Expected output using ``ABCD-abcd``

          Enter your name: ABCD-abcd
          65 66 67 68 45 97 98 99 100

#. Use ``malloc`` to dynamically allocate memory for an ``int`` array.

   .. code-block:: C

       // 'malloc' allocates memory based on the:
       //       size of the data type * the  number of elements.
       // It then returns the pointer to the start of the allocated memory
       int *int_array = malloc(sizeof(int) * size);

       // This example creates an array of 15 floats:
       int *float_array = malloc(sizeof(float) * 15);

   .. note::
       You do not need to preserve the initial pointer value if you do
       not advance the pointer using int_array++;

#. Add the ``integer`` equivalent of the ASCII char to the correct memory
   location or index allocated by ``malloc``.

   a. You need to advance the pointer or get the offset to the
      next memory location.

      .. code-block:: C

        /**
         * There are two ways to move a pointer to the next memory location:
         * 1. Advance the pointer:
         *    int_array++;
         * 2. Use the offset.
         *     *(int_array + i)
         */

          for (int i = 0; i < size; i++)
          {
              // 1. get value from the char array
              // 2. Write that value to the memory address of the pointer
          }

#. Return the pointer to ``int_array``.

Task 4: Print the Array
========================
We still have the problem that we cannot determine programmatically
the size of a dynamically created array. We have to keep track of the
number of elements that we added to it.

Think about how you might solve the problem. Here are some suggestions:

  a. Use a pointer to hold the number of elements added to the array.
     The function sets the value in memory using the pointer, like what we did
     in Lab 4, :ref:`fibonacci-task2` to keep track of ``previous``.
  #. Keep track of the array size in ``main()`` and change the function
     prototype of:

     .. code-block:: C

         // Change:
         int *convert_ascii_to_int(char *str_array) { }

         // to:
         int *convert_ascii_to_int(char *str_array, int size) { }

#. Create a function with this prototype that prints the ``int_array``.

   .. code-block:: C

       /**
        * Prints an array of integer values
        *
        * int array:  the pointer to the array
        * int size:  the size of the array
        */
       void print_array(int *int_array, int size);

#. Print the values from the array

   .. code-block:: text
       :caption: Expected Results using ``fgets()``

       Enter your name: ABCD-abcd
       65 66 67 68 45 97 98 99 100 10

       Enter your name: ABCD 1234 5 6 7 8 9
       65 66 67 68 32 49 50 51 52 32 53 32 54 32 55 32 56 32 57 10

       Enter your name: Jimmy the Memory Leaker
       74 105 109 109 121 32 116 104 101 32 77 101 109 111 114 121 32 76 101 97 107 101 114 10

#. Use an |ASCII Converter| to verify the results

Task 5: Free the Memory
========================
Our program works without a memory leak because it exits immediately
after printing the data in the array. We will have a memory leak
if we repeat the process because we do not release data before
destroying the pointer.

#. Create a loop to prompt the user for a name until they quit.
#. Use ``void free ( void * ptr );`` to release the memory

   - |C Reference function free()|
   - |C - Memory Management|
   - |C Functions - malloc(), calloc(), realloc(), and free()|

.. code-block:: text
    :caption: Example output

    ----------------------------------------------------------------------
    Welcome to our ASCII to Integer converter!
    Enter text up to 64 characters and we'll give the integer equivalent.
    Enter '~' to exit

    Enter some text: abcd
    97 98 99 100 10
    Enter some text: 1
    49 10
    Enter some text: 2
    50 10
    Enter some text: ABCDEFG
    65 66 67 68 69 70 71 10
    Enter some text: `fgets()` works with spaces!
    96 102 103 101 116 115 40 41 96 32 119 111 114 107 115 32 119 105 116 104 32 115 112 97 99 101 115 33 10
    Enter some text: Notice the \n char (0xA)
    78 111 116 105 99 101 32 116 104 101 32 92 110 32 99 104 97 114 32 40 48 120 65 41 10
    Enter some text: ~ Goodbye!
    126 32 71 111 111 100 98 121 101 33 10

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/c/5/index.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
