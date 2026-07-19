********************************
Lab 6: Dynamic Data Structures
********************************

.. include:: ../../includes/prolog.inc

.. include:: ../c-urls.rst

.. contents:: Table of Contents

Overview
==========
You will write a program to dynamically add monthly weather data
to a linked list. A |Linked List| is a dynamic data structure that
can grow in size. Each linked list element contains a ``struct`` with
monthly weather data.

This presentation on |C Dynamic Data Structures| provides a solid overview
of the topics covered today. Please take a few moments to view the slides.

You will learn about:

  - Header (``.h``) files
  - Linked Lists
  - Structs
  - See practical examples of how to use pointers and ``malloc``.

Header Files
--------------------------
A header file (``.h`` file) in C is **an interface file**.

.. important::
    Let's repeat it for emphasis:

    A header file (``.h`` file) in C is an interface file that includes
    function prototypes. It does not contain program logic.

C programs have two main components:

  #. An **INTERFACE** of the library or programs functions that are stored
     header files, such as mylib.h; linkedlist.h
  #. An **IMPLEMENTATION** of the library, functions, or other program logic
     that are stored in code or logic files, such as mylib.c; linkedlist.c

**The program logic** is in the .c source file. This is the code that the
program uses to accomplish some specific task.

**The header files** contain the function prototypes, that is, just
a declaration of which functions can be found in the source file.
The header files also act as the public API to the library. The source
code is not necessary to use a library, but the developer must have the
header file that lists the available functions of that library.

.. tip::
    Click on ``#include <stdio.h>`` in a .c file and press F12 in
    Visual Studio Code.

    Notice that view all of the function prototypes and data structures.

Unfortunately, there are a lot of |bad examples online about header files|.
Notice that the ``.h`` file in the site contains declarations and
program logic. The author of this site does not understand the purpose
of header files. They treat a header file as a regular code file.
Renaming it as .C would be more appropriate.

Here is an excellent example of using |C Header Files| to split your
program into multiple files to create an interface to access the
program's functionality. Notice that the ``.h`` file includes:

  - the necessary includes for the library, such as ``#include <math.h>``
  - defines for constants, such a PI

**Resources**

  - |C Header Files|
  - |How to Create C Libraries|

Structs
--------------------------
A |C struct| is a simple user-defined data structure. It allows you to
encapsulate or serialize data to a single type.

**Resources**

  - |C struct|
  - |C Advanced Data Types|

Linked Lists
-------------------------------
One disadvantage of using arrays to store data is that arrays are static
structures, and therefore cannot be easily extended or reduced to fit
the data set. Also, it is expensive to insert or remove data from any
part of the array except the very end.

A |linked list| is a linear data structure where each element is a
separate object that points to the next object, but the order is
not given by their physical placement in memory. This structure allows
for efficient insertion or removal of elements from any position in
the sequence during iteration because the list points to a memory
address. There are different |Types of Linked Lists|.

**Resources**

  - |linked list|
  - |Types of Linked Lists|
  - |Linked List Operations|
  - |Singly Linked List Tutorial|
  - |Linked List (Wikipedia)|


Task 1: Create a .h File
==========================================
A ``.h`` file should contain definitions only. It should ``NOT`` contain
any code that performs some action.

#. Create a project folder on your computer called ``weather``.
#. Create a header filed called ``weather.h``.
#. Add a check to prevent a duplicate declaration

   .. code-block:: C

       #ifndef WEATHER_H_   /* Include guard */
       #define WEATHER_H_

       // Declarations will go here

       #endif // WEATHER_H_

#. Our library will use ``stdlib.h``. Add it now to the .h file. All files
   that include ``weather.h`` also include ``stdlib.h``. It prevents
   duplication of includes.
#. Create a ``struct`` with the name of ``node``.

   .. code-block:: C

       /**
        * Defines the structure of weather data
        */
       struct node
       {
           char unit;
           int duration;
           int low_temp;
           int high_temp;
           struct node *next;
       };

#. We don't want to refer to a node as ``struct node`` for every reference.
   So, create a new type named ``weather`` as a pointer to ``struct node``.

   .. code-block:: C

       //Define 'weather' as pointer of data type 'struct node'
       typedef struct node *weather;

#. Add the function prototypes for our linked list.

   .. code-block:: C

       /**
        * Creates a new weather node and allocate the required memory
        *
        *   returns:
        *     - An empty linked list node of type weather
        */
       weather create_node();

       /**
        * Function: add_node
        * -------------------------------
        *   Adds a new node to the end of the linked list that contains
        *   weather data for the duration.
        *
        *   weather head:  The head of the linked list
        *   char unit:     The weather units ('c' or 'f')
        *   int duration:  The duration (01-12 for months Jan-Dec; or weeks 01-52, etc.)
        *   int low_temp   The average temp for the duration
        *   int high_temp  The average high temp of the duration
        *
        *   returns:
        *     - The head of the linked list (not the last node)
        */
       weather add_node(weather, char, int, int, int);

You should have a completed header file that defines our type and
the available functions.

Task 2: Create the Library File
==========================================
Now, you can create a library file. You already know which functions
it will include.

#. Create a new file called ``weather.c``.
#. The functions need access to the declarations in the header file.
   Add an include statement for ``weather.h``.

#. Add function ``create_node()`` to create and initialize a new node.

   .. note:: This node exists in memory only. It has not been added
       to the list.

   .. code-block:: C

       /**
        * Function: create_node
        * -------------------------------
        * Creates a new weather node and allocate the required memory
        *
        *   returns:
        *     - An empty linked list node of type weather
        */
       weather create_node()
       {
           // allocate memory using malloc()
           weather new_node = (weather)malloc(sizeof(struct node));

           // Initialize the pointer to the next node as NULL
           new_node->next = NULL;

           return new_node;
       }

#. Add function ``add_node()`` to add a node **to the end of the list**
   that contains data to the linked list.

   .. code-block:: C

       /**
        * Function: add_node
        * -------------------------------
        *   Adds a new node to the end of the linked list that contains
        *   weather data for the duration.
        *
        *   weather head:  The head of the linked list
        *   char unit:     The weather units ('c' or 'f')
        *   int duration:  The duration (01-12 for months Jan-Dec; or weeks 01-52, etc.)
        *   int low_temp   The average temp for the duration
        *   int high_temp  The average high temp of the duration
        *
        *   returns:
        *     - The head of the linked list (not the last node)
        */
       weather add_node(weather head, char unit, int duration, int low_temp, int high_temp)
       {
           // Create a new node
           weather new_node = create_node();

           // Assign the values
           new_node->unit = unit;
           new_node->duration = duration;
           new_node->low_temp = low_temp;
           new_node->high_temp = high_temp;

           // Assign first node to head if the list is empty
           if (head == NULL)
           {
               head = new_node;
           }
           else
           {
               // Get the reference to head so that we can traverse the list
               // without changing the position of the head node
               weather current_node = head;

               // Traverse the list searching for the lastnode
               while (current_node->next != NULL)
               {
                   // Set the current node as next
                   // The last node always points to NULL.
                   current_node = current_node->next;
               }

               // Link the new_node to the node at the end of the list
               current_node->next = new_node;
           }

           // Always return the head node
           return head;
       }


Task 3: Create a Node
==================================
Now, you are ready to use your library to store data in a linked list.

.. note::
    A linked list must always keep the reference to the ``head`` node.
    Otherwise, there is no way to traverse the list and access the data.

#. Create a file called ``main.c`` using the :ref:`basic-c-template`.
#. Include ``weather.h`` so that you can use the linked list.
#. Create the reference to ``head``. You will use this variable anytime
   you use the linked list.

   .. code-block:: C

       // Create a reference to the head node
       weather head = NULL;

#. Create a new linked list by adding a weather node. Use a weather site,
   such as https://www.worldweatheronline.com/oskemen-weather-averages/east-kazakhstan/kz.aspx
   to get the data.

   .. note:: Set the first node as ``head``.

   .. code-block:: C

       char unit = 'c';
       int duration = ;
       low_temp = ;
       high_temp = ;

       head = add_node(head, unit, duration, low_temp, high_temp);

   - Evaluate the code in ``add_node()`` and think why it works when ``head`` is ``NULL``.

#. Print the contents of ``head``.

   .. note::
       Recall that ``weather`` is a pointer to ``node``.
       Therefore, you need to use ``->`` to access the members:

       .. code-block:: C

           int low_temp = head->low_temp;

#. Now, add a second and third month to the list.

   .. note::
       - ``add_node()`` always returns ``head``. So, adding a new node is easy.
       - Use the same method as you did for the first month.

#. Print the contents of the new nodes. Even though ``add_node()``
   returns the head, it also contains the reference to the next node.

   .. code-block:: C

       weather second = head->next;
       weather third = second->next;

Task 4: Add a New ``struct`` Member
=======================================
You've seen how to add elements to the linked list. Now, you will add
a field to the struct to extend the weather data to include the
monthly average. The monthly average is simply the average of the
high and low temps.

#. Add a new field called ``float avg_temp;`` to the struct.
#. Add the new field to the appropriate references in the linked list code.
   You have several options on how to accomplish this:

   a. Calculate the result in ``weather.c`` when you set the data in
      the struct.
   #. Calculate the result in ``main.c`` and pass the value in ``add_node()``.

#. Modify your code to print the monthly average.

Task 5: Print the Data
==========================
Lastly, you should print the results in a nicely formatted table
that displays the values in C or F.

#. Create a function using this prototype that prints the contents
   of the linked list in a tabular list.

   .. code-block:: C

       /**
        * Function: print_weather_data
        * -------------------------------
        * Prints the weather data in the linked list in C or F.
        *
        *   weather head:  The head of the linked list
        *   char unit:     The weather units (C or F)
        */
       void print_weather_data(weather, char);

#. Call the function from ``main()`` to test both units, like this:

   .. code-block:: C

       print_weather_data(head, 'C');
       print_weather_data(head, 'F');

   .. code-block:: text
       :caption: Expected Output (partial set)

       Month   Low    High   Average  (in C)
       -----  -----   ----   -------
         1    -13     -6    -9.5
         2    -12     -3    -7.5
         3     -6     -2    -4.0

       Month   Low    High   Average  (in F)
       -----  -----   ----   -------
         1     8.6    21.2    14.9
         2    10.4    26.6    18.5
         3    21.2    28.4    24.8

#. Use this code to speed up your data entry process.

   .. note::
       - Recall: An array is a pointer.
       - The function does not need to return the array.
       - You can access the array outside of the function once it has
         populated.

   .. code-block:: C

       /**
        * Populates a 2D array of size [n][2] with weather data
        *
        * int (*array)[2]: A pointer to a 2D int array
        *                  {duration},{high, low}
        *
        * Usage:
        *    // Create an array, then pass the reference to function
        *    int temps [12][2];
        *    populate_array(temps);
        */
       void populate_array(int (*array)[2])
       {
           // January
           array[0][0] = -6 ;      // high temp
           array[0][1] = -13;      // low temp

           array[1][0] = -3;
           array[1][1] = -12;

           array[2][0] = -2;
           array[2][1] = -6;

           array[3][0] = 14;
           array[3][1] = 3;

           array[4][0] = 18;
           array[4][1] = 6;

           // June
           array[5][0] = 23;
           array[5][1] = 12;

           array[6][0] = 27;
           array[6][1] = 15;

           array[7][0] = 29;
           array[7][1] = 15;

           array[8][0] = 20;
           array[8][1] = 8;

           array[9][0] = 12;
           array[9][1] = 3;

           array[10][0] = -4;
           array[10][1] = -12;

           // December
           array[11][0] = -5;
           array[11][1] = -12;
       }

#. Loop through the array to create the linked list dynamically
#. Print the linked list data using a ``while`` loop.

   .. code-block:: C

       // Set the pointer to the head in a new variable.
       // Otherwise, you will advance the head and lose the reference.
       weather p = head;

       while (p != NULL)
       {
           // Use p here

           // Move p to next
           p = p->next;
       }

   .. code-block:: text
       :caption: Expected Output (Full data set)

       Month   Low    High   Average  (in C)
       -----  -----   ----   -------
         1    -13     -6    -9.5
         2    -12     -3    -7.5
         3     -6     -2    -4.0
         4      3     14     8.5
         5      6     18    12.0
         6     12     23    17.5
         7     15     27    21.0
         8     15     29    22.0
         9      8     20    14.0
        10      3     12     7.5
        11    -12     -4    -8.0
        12    -12     -5    -8.5

       Month   Low    High   Average  (in F)
       -----  -----   ----   -------
         1     8.6    21.2    14.9
         2    10.4    26.6    18.5
         3    21.2    28.4    24.8
         4    37.4    57.2    47.3
         5    42.8    64.4    53.6
         6    53.6    73.4    63.5
         7    59.0    80.6    69.8
         8    59.0    84.2    71.6
         9    46.4    68.0    57.2
        10    37.4    53.6    45.5
        11    10.4    24.8    17.6
        12    10.4    23.0    16.7

What's Next?
==============
We only implemented a single functionality of a linked list, which
was ``add_last``. There are more |Linked List Operations| to implement.

Here is a list of common linked list operations:

  #. ``add_first``
  #. ``add_last``
  #. ``insert_before``
  #. ``insert_after``
  #. ``delete``
  #. ``clone``

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/c/6/index.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
