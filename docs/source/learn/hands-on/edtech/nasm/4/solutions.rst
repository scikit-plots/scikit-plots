*********************
Lab 4: Solutions
*********************

There is a lot of whitespace between the solutions. Use the menu to
jump to the solution of interest.

.. include:: ../../includes/prolog.inc

.. contents:: Table of Contents

.. _lab_4_1_solution:

Lab 4.1 Solutions
====================

Inverse Triangle
-------------------
This solution is simple because it uses the ``loop`` command.

.. code-block:: asm
    :caption: Simple inverse triangle
    :linenos:

        ; Start instructions
        ; ---------------------------------------------
        ; Register information
        ; EAX:      ; temp var, the character to print
        ; EBX:      ; column counter (inner loop)
        ; ECX:      ; row counter (outer loop)
        ; EDX:      ; not used
        ; ---------------------------------------------
        mov     ecx, 5          ; ECX is i (this must be ECX for 'loop' instruction)
    outer:
        mov     ebx,  0         ; reset j
      inner_start:
        cmp     ebx,  ecx       ; compare j and i
        jge     inner_end       ; if (j => i) exit inner loop
        inc     ebx             ; j++
        mov     eax,  ebx       ; print the counter (or mov a DX label here)
        call    print_int       ; Print j
        jmp     inner_start     ; continue inner loop
      inner_end:
        call    print_nl
        loop    outer           ; if EXC != 0, continue outer loop
        ; End instructions, clean up

.. code-block:: text
    :caption: Expected Output

    12345
    1234
    123
    12
    1

Right Triangle
-------------------

.. code-block:: asm
    :caption: Right triangle
    :linenos:

        ; Start instructions
        ; ---------------------------------------------
        ; Register information
        ; EAX:      ; column loop counter; the character to print
        ; EBX:      ; row loop control variable
        ; ECX:      ; row loop counter
        ; EDX:      ; column loop control variable
        ; ---------------------------------------------

        mov     ecx, 0          ; ECX is i, row counter
        mov     ebx, 5          ; ebx loop condition (number of rows)
    row:
        cmp     ecx,  ebx       ; compare i and number of rows
        jg      row_end         ; if (i > rows) exit loop

        ; equilateral triangle
        mov    edx,  ecx        ; set column control var equal to the row

        mov     eax,  0         ; reset column counter
        call    column          ; print the column data
        call    print_nl
        inc     ecx             ; i++
        jmp     row             ; goto row start
    row_end:
        jmp     end             ; all done

    ; print the column data
    column:
        cmp     eax,  edx       ; compare j and i
        jge     return          ; if (j => i) call "ret"
        inc     eax             ; j++
        call    print_int       ; Print loop counter in EAX
        jmp     column          ; continue column loop

    return:
        ret                     ; return to the last address called

    end:
        ; End instructions, clean up


.. code-block:: text
    :caption: Expected Output

    1
    12
    123
    1234
    12345

*Scroll down for next solution:*

|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|
|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|

.. _lab_4_2_solution:

Lab 4.2 Solutions
====================

Wide Triangle
---------------

.. code-block:: asm
    :linenos:

        ; Start instructions
        ; ---------------------------------------------
        ; Register information
        ; EAX:      ; column loop counter; the character to print
        ; EBX:      ; row loop control variable
        ; ECX:      ; row loop counter
        ; EDX:      ; column loop control variable
        ; ---------------------------------------------

        mov     ecx,  0         ; ECX is i, row counter
        mov     ebx,  5         ; ebx loop condition (number of rows)
    row:
        cmp     ecx,  ebx       ; compare i and number of rows
        jg      row_end         ; if (i > rows) exit loop

        ; wide triangle
        mov    eax,   3         ; multiplication factor
        mul    ecx              ; multiple the factor (EAX) by the loop counter in ECX
                                ; to print a wide triangle
        mov    edx,   eax       ; put the number of columns to print in EDX

        mov     eax,  0         ; reset column counter
        call    column          ; print the column data
        call    print_nl
        inc     ecx             ; i++
        jmp     row             ; goto row start
    row_end:
        jmp     end             ; all done

    ; print the column data
    column:
        cmp     eax,  edx       ; compare j and i
        jge     return          ; if (j => i) call "ret"
        inc     eax             ; j++
        push    eax             ; save counter
        mov     eax,   char     ; get character to print
        call    print_string    ; Print *
        pop     eax             ; restore counter
        jmp     column          ; continue column loop

    return:
        ret                     ; return to the last address called

    end:
        ; End instructions, clean up

.. code-block:: text
    :caption: Wide Triangle Expected Results (i * 3)
    :linenos:

    *
    * * * *
    * * * * * * *
    * * * * * * * * * *
    * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *

*Scroll down for next solution:*

|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|
|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|

.. _lab_4_3_solution:

Lab 4.3 Solutions
====================

Square Curve
---------------

.. code-block:: asm
    :linenos:

        ; Start instructions
        ; ---------------------------------------------
        ; Register information
        ; EAX:      ; column loop counter; the character to print
        ; EBX:      ; row loop control variable
        ; ECX:      ; row loop counter
        ; EDX:      ; column loop control variable
        ; ---------------------------------------------

        mov     ecx,  0         ; ECX is i, row counter
        mov     ebx,  6         ; ebx loop condition (number of rows)
    row:
        cmp     ecx,  ebx       ; compare i and number of rows
        jg      row_end         ; if (i > rows) exit loop

        ; Square curve
        mov     eax, ecx        ; Get the row
        mul     ecx             ; EAX = row*row
        mov     edx, eax        ; put the number of columns to print in EDX

        mov     eax,  0         ; reset column counter
        call    column          ; print the column data
        call    print_nl
        inc     ecx             ; i++
        jmp     row             ; goto row start
    row_end:
        jmp     end             ; all done

    ; print the column data
    column:
        cmp     eax,  edx       ; compare j and i
        jg      return          ; if (j => i) call "ret"
        inc     eax             ; j++
        push    eax             ; save counter
        mov     eax,   char     ; get character to print
        call    print_string    ; Print *
        pop     eax             ; restore counter
        jmp     column          ; continue column loop

    return:
        ret                     ; return to the last address called

    end:
        ; End instructions, clean up


.. code-block:: text
    :caption: Square curve (n * n) Expected Results
    :linenos:

    *
    * *
    * * * * *
    * * * * * * * * * *
    * * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

*Scroll down for next solution:*

|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|
|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|

.. _lab_4_4_solution:

Lab 4.4 Solutions
====================

Skinny Triangle
-------------------

.. code-block:: asm
    :linenos:

        ; Start instructions
        ; ---------------------------------------------
        ; Register information
        ; EAX:      ; column loop counter; the character to print
        ; EBX:      ; row loop control variable
        ; ECX:      ; row loop counter
        ; EDX:      ; column loop control variable
        ; ---------------------------------------------

        mov     ecx,  1         ; ECX is i, row counter
        mov     ebx,  21        ; ebx loop condition (number of rows)
    row:
        cmp     ecx,  ebx       ; compare i and number of rows
        jg      row_end         ; if (i > rows) exit loop

        ;
        ; skinny triangle   ** ECX must start at 1, not 0. **
                                ; EAX = EAX/ECX; columns = rows/x
        mov     eax,  ecx       ; move the current row value into EAX
        push    ecx             ; save current row value
        mov     ecx,  5         ; put our divisor in ECX
        mov     edx,  0         ; initialize the EDX (where the remainder goes)
        div     ecx             ; EAX = EAX / ECX
        mov     edx,  eax       ; put the number of columns to print in EDX
        pop     ecx             ; restore the row value to ECX
        ;
        mov     eax,  0         ; reset column counter
        call    column          ; print the column data
        call    print_nl
        inc     ecx             ; i++
        jmp     row             ; start new row
    row_end:
        jmp     end             ; all done

    ; print the column data
    column:
        cmp     eax,  edx       ; compare j and i
        jg      return          ; if (j => i) call "ret"
        inc     eax             ; j++
        push    eax             ; save counter
        mov     eax,   char     ; get character to print
        call    print_string    ; Print *
        pop     eax             ; restore counter
        jmp     column          ; continue column loop

    return:
        ret                     ; return to the last address called

    end:
        ; End instructions, clean up

.. code-block:: text
    :caption: The Burj Khalifa! n/5

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

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/nasm/4/solutions.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
