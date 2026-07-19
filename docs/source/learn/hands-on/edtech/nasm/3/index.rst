*****************************
Lab 3: Control Structure
*****************************

.. include:: ../../includes/prolog.inc

.. include:: ../asm-urls.rst

.. contents:: Table of Contents

Overview
==========
Control structures (branches and loops) are necessary in programming.

You will learn how to make if/else blocks and loops.

.. note::
    These examples use labels that are familiar for control statements.
    Your program should not these types of labels. Your should use
    descriptive labels that describe the purpose of the block.

    These types of labels will not be accepted:

    .. code-block:: text

        loop
        end_if
        zzzzzz
        asdf
        does_something

Sources
----------

- |NASM Control Structures (Module)| - University of Hawaii
- |Jumps and Branches| - University of Hawaii Presentation
- |Control Structures| - University of Hawaii Presentation
- |Conditionals: Goto and Branch Instructions| - University of Alaska

Branches
===========
Branches, go statements, or jumps usually happen after a comparison
or operation that can set a flag on the processor.

They are the *conditional* version of the unconditional ``jmp`` command.

  .. code-block:: asm

          mov  eax,  100
          jmp  next   ; jump to next
          ; skip these commands
          add  eax,  ecx
      next:
          ; do something here


Branch on State of Flag
-----------------------------

.. list-table:: NASM Jump Instructions (Generic)
   :header-rows: 1

   * - Instruction
     - Condition
     - Info

   * - JZ
     - Zero flag: |br| Branches if ZF is set
     - .. code-block:: asm

           mov  edx, 5
           mov  ecx, 5
           sub  ecx, edx
           ; jump if result == 0
           jz   zero_value
   * - JNZ
     - Zero flag: |br| Branches if ZF is unset
     - .. code-block:: asm

           mov  ebx, 5
           cmp  ebx, 0
           ; jump if ebx != 0
           jnz   value_not_zero
   * - JO
     - Signed overflow: |br| Branches if OF is set
     - .. code-block:: asm

         mov  ebx, 2147483647
         add  ebx, 1
         jo  overflow_detected
         ; FLAGS = 0A96 OF    SF    AF PF
   * - JNO
     - Signed overflow: |br| Branches if OF is unset
     - .. code-block:: asm

         mov  ebx, 2147483647
         sub  ebx, 1
         jno  no_overflow_detected
         ; FLAGS = 0202
   * - JS
     - Sign flag: |br| Branches is SF is set
     - .. code-block:: asm

         mov  ebx, 0
         sub  ebx, 10
         js  negative_value
         ; FLAGS = 0297       SF    AF PF CF
   * - JNS
     - Sign flag: |br| Branches is SF is unset
     - .. code-block:: asm

         mov  ebx, 10
         sub  ebx, 10
         jns  not_negative
         ; FLAGS = 0246          ZF    PF
   * - JC
     - Unsigned Carry flag |br| branches if CF is set
     - .. code-block:: asm

         mov  eax, 4294967294
         add  eax, 2
         jc   handle_unsigned_carry
         ; FLAGS = 0257          ZF AF PF CF
   * - JNC
     - Unsigned Carry flag |br| Branches if CF is unset
     - .. code-block:: asm

         mov  eax, 4294967294
         add  eax, 1
         jnc  no_unsigned_carry
         ;FLAGS = 0286       SF       PF

Branches for Equality Tests
----------------------------
This group is a result of command ``cmp``.

  .. code-block:: asm

      cmp eax, ecx    ; compare and then evaluate

.. list-table:: NASM Jump Instructions (Signed)
   :header-rows: 1

   * - Instruction
     - Condition
     - Info
   * - JE
     - Jump if ``cmp`` is equal
     - .. code-block:: asm

           mov  eax, 10
           mov  ecx, 10
           cmp  eax, ecx
           ; jump if eax == ecx
           je   match_found
   * - JNE
     - Jump if ``cmp`` is not equal
     - .. code-block:: asm

           mov  eax, 5
           mov  ecx, 10
           cmp  eax, ecx
           ; jump if eax != ecx
           jne  not_equal
   * - JG
     - Signed  ``>``
     - greater than
   * - JGE
     - Signed  ``>=``
     - greater than or equals
   * - JL
     - Signed  ``<``
     - less than
   * - JLE
     - Signed  ``<=``
     - less than or equals
   * - JA
     - Unsigned ``>``
     - above
   * - JAE
     - Unsigned ``>=``
     - above or equals
   * - JB
     - Unsigned  ``<``
     - below
   * - JBE
     - Unsigned  ``<=``
     - below or equals

Consider the following C-like code wither register-like variables

  .. code-block:: c

      if (EAX == 0)
         EBX = 1;
      else
         EBX = 2;

Here it is in x86 assembly. Notice the ``jz`` flag.

  .. code-block:: asm

          cmp    eax,  0       ; do the comparison
          jz     then_block    ; if = 0, then goto then_block
          mov    ebx,  2       ; else clause
          jmp    next          ; jump over the then clause
      then_block:
          mov    ebx,  1       ; then clause
      next:
          ; continue the program


Control Block Examples
==========================

If-Else block
------------------------
View runnable file: |if-then.asm.txt|

.. code-block:: C
    :caption: Generic ``if`` statement

    if (condition) then
        then_block

.. code-block:: ASM
    :caption: NASM implementation of ``if`` statement

        ; instructions to set flags (e.g., cmp ...)
        jXX  end_if         ; evaluate flag. Jump on true condition

        ; then_block code

    end_if:
        ; program continues


If-Then-Else block
------------------------
View runnable file: |if-then-else.asm.txt|

.. code-block:: C
    :caption: Generic ``if-then-else`` statement

    if (condition) then
        then_block
    else
        else_block

.. code-block:: ASM
    :caption: NASM implementation of ``if-then-else`` statement

        ; instructions to set flags (e.g., cmp ...)
        jXX    else_block       ; evaluate flag. Jump on true condition

        ; then_block code
        jmp end_if              ; skip the else_block

    else_block:

        ; code for the else block

    end_if:
        ; program continues


For loops
------------------------
There are two variants of the ``for`` loop. The first is a
traditional approach incrementing ``i``. The second option
uses the ``loop`` command, which operates in reverse order.

Traditional ``for`` loop
~~~~~~~~~~~~~~~~~~~~~~~~~~
View runnable file: |for.asm.txt|

.. code-block:: C
    :caption: Generic ``for`` loop

    sum = 0;
    for (i = 0; i <= 10; i++) {
        sum += i;
    }

.. code-block:: ASM
    :caption: NASM implementation of a ``for`` loop

        mov    eax,  0          ; eax is sum
        mov    ebx,  0          ; ebx is i (could be ecx or edx)
    loop_start:
        cmp    ebx,  10         ; compare i and 10
        jg     loop_end         ; if (i > 10) goto loop_end
        add    eax, ebx         ; sum += i
        inc    ebx              ; i++
        jmp    loop_start       ; goto loop_start
    loop_end:
        ; program continues

``loop`` command (Reverse loop)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
View runnable file: |loop.asm.txt|

.. code-block:: C
    :caption: Generic reverse ``for`` loop

    sum = 0;
    for (i = 10; i > 0; i--) {
        sum += i;
    }

.. code-block:: ASM
    :caption: NASM implementation of a ``for`` loop

        mov    eax, 0          ; eax is the sum
        mov    ecx, 10         ; ecx is i (this must be ECX)
    loop_start:
        add    eax, ecx        ; sum += i
        loop   loop_start      ; if i > 0 then  go to loop_start

        ; program continues when ECX == 0

While loop
------------------------
View runnable file: |while.asm.txt|

.. code-block:: C
    :caption: Generic ``while`` loop

    while (condition) {
       body
    }

.. code-block:: ASM
    :caption: NASM implementation of a ``while`` loop

    while:
        ; instructions to set flags (e.g., cmp...)
        jXX    end_while        ; evaluate flag. Jump on true condition

        ; body of loop
        jmp    while            ; continue loop

    end_while:
        ; program continues

Do-While loop
------------------------
View runnable file: |do-while.asm.txt|

.. code-block:: C
    :caption: Generic do-while loop

    do {
       body
    } while (condition)

.. code-block:: ASM
    :caption: NASM implementation of a do-while loop

    do:
        ; body of loop
        ; instructions to set flags (e.g., cmp...)
        jXX    do                  ; evaluate flag. Jump on true condition

        ; program continues

Task 1: If Statement
================================================
You are performing system upgrades. However, not all systems will receive
the upgrade at the same time. You need to write code to prevent a system
crash and to protect data from an overflow.

Your task is to create a NASM program that squares a number to determine
if it will fit in the current register. If not, print the original value.
The program must work with any register size (16, 32, or 64 bit).

.. code-block:: C
    :caption: C Template

    int value = 8192;

    if (can_square)
        value = value * value;

    printf(value);

See the :ref:`lab_3_1_solution` if you need help.

Task 2: If/Else Statement
===================================================
Your task is to create a NASM program that determines if a number is
even or odd. Print 0 for even numbers and 1 for odd numbers.

.. code-block:: C
    :caption: C Template

    int value = 8192;

    if (even)
        value = 0;
    else
        value = 1;

    printf(value);

See the :ref:`lab_3_2_solution` if you need help.

Task 3: While Loop
===================================================
You need to track the number of cycles that an operation takes to
complete. Create a while loop that performs some action. Print the
number of loop iterations.

You could evaluate a zero value (ZF), less than, or greater than

.. code-block:: C
    :caption: C Template

    int counter = 0;

    // Example using greater than
    while (value <! maximum) {

       value = value + 10;
       counter ++;
    }

    printf(counter);

See the :ref:`lab_3_3_solution` if you need help.

Task 4: Factorial
===================================================
Use the ASM implementation of a ``for`` loop to calculate the factorial
of a number. See :ref:`C programming lab 1<c-lab-1>` for additional
information.

.. code-block:: C
    :caption: C Template

    int i;                           // ECX
    int factorial = 5;               // EBX
    int result = 1;                  // EAX

    for (i = 1; i <= factorial; i++)
    {
        result = result * i;

        // Check for overflow. -1 will indicate that an error occurred.
        if (OVERFLOW_FLAG) {
            factorial = 0;          // loop condition
            result = -1;            // set to -1 (invalid factorial)
            print(i);               // Record max factorial number + 1
        }

        // Optionally, print the incremental number to help you debug
        // printf(result);
    }

    printf(result);

.. code-block:: text
    :caption: Expected Results

    ; 5!
    120

    ; 12!
    479001600

    ; 18! (n! > 12)
    13          ; 13! caused an overflow
    -1          ; error

See the :ref:`lab_3_4_solution` if you need help.

Solutions
==============================

.. toctree::
   :caption: Lab 3 Solutions

   solutions

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/nasm/3/index.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
