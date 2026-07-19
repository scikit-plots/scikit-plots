***************************************
Lab 4: Nested Loops and Sub Routines
***************************************

.. include:: ../../includes/prolog.inc

.. include:: ../asm-urls.rst

.. contents:: Table of Contents

Overview
===========
Assembly uses goto statements to jump to a label, which is an instruction
at a specific memory location. As a result, loops do not need to be
*nested* in each other as they do in higher level languages. However,
you can put the inner loop in a function. Using the ``call`` instruction
in ASM is similar to calling a function because it preserves the return
point. A ``jmp`` statement does not.

Consider these two C programs that print a right triangle. They do
the same thing. The first example does the code in nested loop but
the second example moves the inner loop to a function. Where it does
the work is not important.

C Nested Loop Examples
------------------------
.. code-block:: c
    :caption: Classic nested loop in C. Run the code at https://ide.judge0.com/
    :linenos:

    #include <stdio.h>

    int main()
    {
        int rows = 5;

        for (int i = 1; i <= rows; i++)
        {
            // Perform work in the inner loop
            for (int j = 1; j <= i; j++) {
                printf("* ");
            }

            // Finish the work of the outer loop
            printf("\n");
        }
        return 0;
    }


.. code-block:: c
    :caption: Inner loop moved to function. Run the code at https://ide.judge0.com/
    :linenos:
    :emphasize-lines: 11

    #include <stdio.h>

    void print_row(int);

    int main()
    {
        int rows = 5;

        for (int i = 1; i <= rows; i++)
        {
            print_row(i);   // Perform the work of the inner loop
            printf("\n");   // Finish the work of the outer loop
        }
        return 0;
    }

    /**
     * Performs the work of the inner loop
     * int i: The counter of the outer loop
     */
    void print_row(int i)
    {
        for (int j = 1; j <= i; j++) {
            printf("* ");
        }
    }

ASM Nested Loop Examples
-------------------------

.. code-block:: asm
    :caption: ASM nested loop
    :linenos:
    :emphasize-lines: 11, 17

    segment .text
    asm_main:                   ; entry point (called from driver.c)
        ; Initialize
        enter 0,0               ; Initialize register EBP
        pusha                   ; Push register contents to the stack
        ; Start instructions
        mov    ecx, 5           ; ECX is i (this must be ECX for 'loop' instruction)

    outer:
        mov    eax,  0          ; reset j
      inner_start:
        cmp    eax,  ecx        ; compare j and i
        jge    inner_end        ; if (j => i) exit inner loop
        inc    eax              ; j++
        call   print_int        ; Print j
        jmp    inner_start      ; continue inner loop
      inner_end:
        call   print_nl
        loop   outer            ; if ECX != 0, continue outer loop

        ; Program ends when ECX reaches 0
        ; End instructions, clean up
        popa                     ; Restore register data from the stack
        mov eax, 0               ; flush EAX of data
        leave                    ; restore the stack
        ret                      ; Return from main back into C library wrapper

We can replicate the functionality by moving the inner loop to another
location in the code. However, the logic becomes fragment. We have to
put in a label inside of the outer loop so that the inner loop knows
where to go after finishing its work.

.. code-block:: asm
    :caption: ASM Loops using jump statements
    :linenos:
    :emphasize-lines: 12,15,23

    segment .text
    asm_main:                   ; entry point (called from driver.c)
        ; Initialize
        enter 0,0               ; Initialize register EBP
        pusha                   ; Push register contents to the stack
        ; Start instructions
        mov    ecx, 5           ; ECX is i (this must be ECX for 'loop' instruction)

    ; start outer loop
    outer_start:
        mov    eax,  0          ; reset j
        jmp    inner_start      ; Perform the work of the inner loop

    ; Do the work of the inner loop
    outer_work:                 ; <--- The inner loop needs this reference
        call   print_nl
        loop   outer_start      ; if ECX != 0, continue outer loop
        jmp    end              ; The work of the loop is done. Jump to end.

    ; do the work of the inner loop
    inner_start:
        cmp    eax,  ecx        ; compare j and i
        jge    outer_work       ; if (j => i) return to outer loop
        inc    eax              ; j++
        call   print_int        ; Print j
        jmp    inner_start      ; continue inner loop

    ; All done
    end:
        ; End instructions, clean up
        popa                     ; Restore register data from the stack
        mov eax, 0               ; flush EAX of data
        leave                    ; restore the stack
        ret                      ; Return from main back into C library wrapper

NASM Subroutines
-------------------
A subroutine (function or method) is used to perform repeat actions to
avoid code duplication and to simplify program logic. We use subroutines
``print_int`` to print an integer to the console. Otherwise, we would
need to insert this code block every time we want to print a value:

.. code-block:: asm
    :caption: Subroutine ``print_int``
    :linenos:

    push    eax
    push    dword int_format
    call    _printf
    pop     ecx
    pop     ecx

Here is the template for a subroutine:

.. code-block:: asm
    :caption: Subroutine template
    :linenos:

    label:
        push   ecx              ; save register X data on stack
        ; do something with ECX (or other register)
        pop    ecx              ; restore data to register X
        ret                     ; return to the address that called the label

.. code-block:: asm
    :caption: ASM Loop using a subroutine.
    :linenos:
    :emphasize-lines: 12,18,20,25

    segment .text
    asm_main:                   ; entry point (called from driver.c)
        ; Initialize
        enter 0,0               ; Initialize register EBP
        pusha                   ; Push register contents to the stack
        ; Start instructions
        mov    ecx, 5           ; ECX is i (this must be ECX for 'loop' instruction)

    ; start outer loop
    outer_start:
        mov    eax,  0          ; reset j
        call   inner_start      ; Perform the work of the inner loop
        call   print_nl
        loop   outer_start      ; if ECX != 0, continue outer loop
        jmp    end              ; The work of the loop is done. Jump to end.

    ; do the work of the inner loop
    inner_start:
        cmp    eax,  ecx        ; compare j and i
        jge    return           ; if (j => i) call "ret"
        inc    eax              ; j++
        call   print_int        ; Print j
        jmp    inner_start      ; continue inner loop
    return:
        ret                     ; return to the last address called (line 12)

    end:

        ; End instructions, clean up
        popa                     ; Restore register data from the stack
        mov eax, 0               ; flush EAX of data
        leave                    ; restore the stack
        ret                      ; Return from main back into C library wrapper


Making (inverse) Right Triangles
===================================================
We all like to create useless programs. :))

Use NASM to create a triangle or similar design using a character,
such as ``*``. You will use a nested ``for`` loop. The width should
not exceed 150 characters to prevent wrapping.

Here is the `C Code <https://www.w3resource.com/c-programming-exercises/for-loop/c-for-loop-exercises-9.php>`_
to help you.

Let's make things more complicated. You can't call ``print_string`` more
than once per loop iteration. It would be too easy to call ``print_string``
twice. Instead, use ``EDX`` to determine when to exit the inner loop.
Use these variables:

Development Tips
--------------------
The simplest implementation is to use the ``loop`` command to print
an inverse triangle (start at the maximum and then count down). See
one of the *ASM Nested Loop Examples* listed above. You can use these
registers.

    .. code-block:: asm
        :caption: Register for using ``loop``

        EAX:      ; temp var, the character to print
        EBX:      ; column counter (inner loop)
        ECX:      ; row counter (outer loop)
        EDX:      ; temp var, loop control variable

Printing the triangle with the base at the bottom is more complicated.
You do not have enough registers to store the data. So, you have to
push the data in one of the registers to the stack and then restore it.

    .. code-block:: asm
        :caption: Registers for using loop control variables

        EAX:      ; column loop counter; the character to print
        EBX:      ; row loop control variable
        ECX:      ; row loop counter
        EDX:      ; column loop control variable

The column loop would look something like this:

    .. code-block:: asm
        :caption: Column code snippet using EAX as counter
        :linenos:
        :emphasize-lines: 5,8

        column:
                                    ; compare j and i
                                    ; if (j => i), jump out of loop
            inc    eax              ; j++
            push   eax              ; save counter
            mov    eax, char        ; get character to print
                                    ; Print char
            pop    eax              ; restore counter
            jmp    column           ; continue column loop

Task 1: Symmetrical Triangle
==============================
Create a symmetrical triangle, such as 3x3 or 5x5. ``EDX ==  ECX``

.. tip::
    Add a space between each ``*`` if you want the height to match the width.

.. code-block:: text
    :caption: Sample right triangle where *n* = 5.
    :linenos:

    *
    * *
    * * *
    * * * *
    * * * * *


See the :ref:`lab_4_1_solution` if you need help.

Task 2: Fat Triangle
==============================
Modify the loop variables to create more columns than rows. The slope
of the triangle should consistent. ``EDX == (ECX * 2)`` or
``EDX == (ECX * 3)`` are examples.

.. code-block:: text
    :caption: i * 2; Reverse loop

    5  * * * * * * * * * *
    4  * * * * * * * *
    3  * * * * * *
    2  * * * *
    1  * *
    0  *

See the :ref:`lab_4_2_solution` if you need help.

Task 3: Curves
==============================
Create a regular or inverse curve from squaring numbers *(n*n)*.

For a challenge, create power table (2 :sup:`n`). The power curve is
much harder because either (1) you have to keep track of the last result
by storing it using a ``RESX`` label to square or (2), you can create
a subroutine.

.. note::
    Recall that ``mul ecx`` uses both EAX and EDX.

.. code-block:: text
    :caption: Square curve (n * n)
    :linenos:

    *
    **
    *****
    **********
    *****************
    **************************
    *************************************
    **************************************************
    *****************************************************************


.. code-block:: text
    :caption: Power curve
    :linenos:

    *
    **
    ****
    ********
    ****************
    ********************************
    ****************************************************************

See the :ref:`lab_4_3_solution` if you need help.

Task 4: Skinny Triangle
==============================
Try a more challenging triangle by modify the loop variables to
create more rows than columns using the ``div`` command.

.. code-block:: asm
    :caption: Skinny triangle column calculator

                            ; EAX = EAX/ECX; columns = rows/x
    mov     eax, ecx        ; move the current row value into EAX
    push    ecx             ; save current row value on the stack
    mov     ecx, 5          ; put our divisor in ECX
    mov     edx, 0          ; initialize EDX (where the remainder goes)
    div     ecx             ; EAX = EAX / ECX
    mov     edx, eax        ; put the number of columns to print in EDX
    pop     ecx             ; restore the row value from the stack to ECX

.. code-block:: text
    :caption: The Burj Khalifa! n/5
    :linenos:

    *
    *
    *
    *
    **
    **
    **
    **
    **
    ***
    ***
    ***
    ***
    ***
    ****
    ****
    ****
    ****
    ****
    *****
    *****

See the :ref:`lab_4_4_solution` if you need help.

Solutions
==============================

.. toctree::
   :caption: Lab 4 Solutions

   solutions

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/nasm/4/index.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
