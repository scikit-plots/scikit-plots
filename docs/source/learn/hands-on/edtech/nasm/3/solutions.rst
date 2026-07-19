*********************
Lab 3: Solutions
*********************

There is a lot of whitespace between the solutions. Use the menu to
jump to the solution of interest.

.. include:: ../../includes/prolog.inc

.. contents:: Table of Contents

.. _lab_3_1_solution:

Lab 3.1 Solutions
====================

.. code-block:: text
    :caption: ASM If Condition
    :linenos:

    ; ASM Simple IF statement of this if block
    ;
    ; // Value to try to square
    ; int value = 8192;
    ;
    ; // If successful, set the value
    ; if (can_square)
    ;     value = value * value;
    ;
    ; // Print the value, either original or modified
    ; printf(value);
    ;

    ; include directives
    global asm_main
    %include "../lib/asm_io.inc"

    segment .data
        ; DX directives (read only, static data)

    segment .bss
        ; RESX directives (writable data)

    segment .text
    asm_main:                   ; entry point (called from driver.c)
        ; instructions

        enter 0,0               ; Initialize register EBP
        pusha                   ; Push register contents to the stack

        ; Starting values
        mov     ecx,  8192      ; Result OK.         Print ---> 67108864
      ; mov     ecx,  65536     ; Overflow detected. Print ---> 65536
        mov     eax,  ecx       ; Initialize EAX

        ; Try to square the value. Then, evaluate for OF flag
        mul     eax             ; value = value * value

        ; Verify flags with registers
        dump_regs 0

    ; Here, we have to check for the overflow flag. This is the only way to know
    ; if the value can be squared. We have to look at it AFTER the operation.
    ;
    ;   if (!can_square)
    ;      EAX = ECX;           ; put original value in EAX (can't square it)
    ;
        jno      end            ; no overflow, value is ok. Go to end
        mov     eax,    ecx     ; overflow happened, move ECX back to EAX to print

    end:

        ; Print the value in EAX. It will be the square or the original
        call    print_int        ; print the value in EAX
        call    print_nl         ; print in new line

        popa                     ; Restore register data from the stack
        mov eax, 0               ; flush EAX of data
        leave                    ; restore the stack
        ret                      ; Return from main back into C library wrapper


*Scroll down for next solution:*

|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|
|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|

.. _lab_3_2_solution:

Lab 3.2 Solutions
====================

.. code-block:: text
    :caption: Determines if Number is Even or Odd
    :linenos:

    ; IF/Else Block
    ;
    ; Evaluates a number for even or odd.
    ; Print 0 if the number is even
    ; Print 1 if the number is odd.
    ;
    ; include directives
    global asm_main
    %include "../lib/asm_io.inc"

    segment .data
        ; DX directives (read only, static data)

    segment .bss
        ; RESX directives (writable data)

    segment .text
    asm_main:                   ; entry point (called from driver.c)
        ; instructions

        enter 0,0               ; Initialize register EBP
        pusha                   ; Push register contents to the stack

        ; Starting values
        mov     eax,  10        ; Even case
      ; mov     eax,  9         ; Odd case
        mov     edx,  0          ; Initialize EDX for division

        ; Divide the number and check for remainder
        mov     ecx,  2         ; the divisor
        div     ecx             ; divide and check EDX for a remainder

        ; evaluate registers for EDX
        dump_regs 0

        cmp     edx,  0         ; compare, and then jump
        jnz     odd_number      ; JNZ: Jump if not zero. EDX has a remainder

        ; Our number is even (EDX was 0)
        mov     eax,  0         ; --> 0 means even number
        jmp     print           ; jump over the else block to end it

        ;The number is odd
    odd_number:
        mov     eax, 1           ; value = 1

    print:
        ; print the contents of AEX: 0 or 1
        call    print_int        ; print eax
        call    print_nl         ; print in new line
        ;
        popa                     ; Restore register data from the stack
        mov eax, 0               ; flush EAX of data
        leave                    ; restore the stack
        ret                      ; Return from main back into C library wrapper



*Scroll down for next solution:*

|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|
|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|

.. _lab_3_3_solution:

Lab 3.3 Solutions
====================

.. code-block:: text
    :caption: While loop implementation
    :linenos:

    ; Loop until a number becomes negative
    ;
    ; An implementation of:
    ; while (number >= 0)
    ;

    ; include directives
    global asm_main
    %include "../lib/asm_io.inc"

    segment .data
        ; DX directives (read only, static data)

    segment .bss
        ; RESX directives (writable data)

    segment .text
    asm_main:                   ; entry point (called from driver.c)
        ; instructions

        enter 0,0               ; Initialize register EBP
        pusha                   ; Push register contents to the stack
        ;
        mov     eax,  999999    ; Starting number

        ; debug
        dump_regs  0;           ; current state of registers

    subtract:
        sub      eax, 111110        ; subtract until we get to a negative value
        js       negative_number    ; end loop if SF flag is set
        ; Print current results
        call     print_int      ; print eax
        call     print_nl       ; new line
        jmp      subtract       ; continue loop

    negative_number:
        dump_regs  0;            ; verify SF flag

        ; end
        ;
        popa                     ; Restore register data from the stack
        mov eax, 0               ; flush EAX of data
        leave                    ; restore the stack
        ret                      ; Return from main back into C library wrapper

*Scroll down for next solution:*

|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|
|br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br| |br|


.. _lab_3_4_solution:

Lab 3.4 Solutions
====================

.. code-block:: text
    :caption: ASM Factorial Implementation
    :linenos:

    ; Prints the result of a factorial.
    ; Generates an error on overflow
    ;

    ; include directives
    global asm_main
    %include "../lib/asm_io.inc"

    segment .data
        ; DX directives (read only, static data)

    segment .bss
        ; RESX directives (writable data)

    segment .text
    asm_main:                    ; entry point (called from driver.c)
        ; instructions

        enter 0,0                ; Initialize register EBP
        pusha                    ; Push register contents to the stack
        ;

        mov    ecx,  1          ; ecx is i
        mov    ebx,  5          ; factorial to calculates 5! = 120
      ; mov    ebx,  18         ; Test error handling. Causes overflow
        mov    eax,  1          ; eax is factorial result

    ; Simple "for" loop to calculate the factorial of a number
    factorial:
        cmp    ecx,  ebx        ; compare i and factorial
        jg     print            ; if (i > factorial) exit loop
        mul    ecx              ; result = result * i;
        jo     overflow         ; error on overflow flag
        inc    ecx              ; i++
        jmp    factorial        ; process next number

    ; Encountered an overflow. Handle error
    overflow:
        mov    eax,  ecx        ; Move counter to AEX
        call   print_int        ; print counter (the factorial that caused the OF)
        call   print_nl         ; print nl
        mov    eax,  -1         ; set to invalid result

    print:
        call print_int          ; print factorial
        call print_nl           ; print new line

        ; program continues
        ;
        popa                     ; Restore register data from the stack
        mov eax, 0               ; flush EAX of data
        leave                    ; restore the stack
        ret                      ; Return from main back into C library wrapper

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/nasm/3/solutions.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
