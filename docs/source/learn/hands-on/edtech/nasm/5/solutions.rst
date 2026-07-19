*********************
Lab 5: Solutions
*********************

.. code-block:: text
    :caption: Example using a subprogram to calculate an average
    :linenos:

    ; include directives
    global asm_main
    %include "../lib/asm_io.inc"

    segment .data
        msg_int     db      "Enter an integer: ", 0
        msg_avg     db      "The average of the registers is: ", 0

    segment .bss
        ; RESX directives (writable data)

    segment .text
    asm_main:                   ; entry point (called from driver.c)
        ; instructions

        enter   0,0             ; Initialize register EBP
        pusha                   ; Push register contents to the stack
        ;
        mov     eax, msg_int
        call    print_string    ; Get int
        call    read_int        ; eg: EAX = 256 (0x0100)
        mov     ebx, eax        ; copy to other registers
        mov     ecx, eax
        mov     edx, eax
        shl     ebx, 1          ; perform a left bit shift of 1 place
        shl     ecx, 2          ; perform a left bit shift of 2 places
        shl     edx, 3          ; perform a left bit shift of 3 places
        ;
        call    print_nl
        ;
        ; Example
        ; EAX = 00000100 EBX = 00000200 ECX = 00000400 EDX = 00000800
        dump_regs 0             ; verify registers (debug)
        ;
        call    average         ; Calculates the average, store in EAX
        ;
        ; Registers EBX, ECX, and EDX should be unchanged
        ; EAX contains the average: 256+512+1024+2048 / 4 = 960 (0x03C0)
        ;
        ; EAX = 00000027 EBX = 0000001F ECX = 0000002F EDX = 0000003F
        dump_regs 2             ; verify registers (debug)
        ;
        mov     ecx, eax        ; save the returned value in EAX
        mov     eax, msg_avg    ; Move string to print
        call    print_nl
        call    print_string
        mov     eax, ecx        ; Move avg to EAX to print
        call    print_int
        call    print_nl
        ;
        popa                     ; Restore register data from the stack
        mov     eax, 0           ; flush EAX of data
        leave                    ; restore the stack
        ret                      ; Return from main back into C library wrapper

    ;---------------------------------------------------------------
    ;  FUNCTION: average
    ;  - Sums the values in EAX, EBX, ECX and EDX and then calculates
    ;    the average. The remainder in EDX is discarded.
    ;  - Returns the average returns the result in EAX.
    ;---------------------------------------------------------------

    segment .bss
        returnvalue     resd     1   ; place in memory for the return value

    segment .text
    average:

        push    ebp                     ; save original EBP
        mov     ebp, esp                ; set EBP = ESP
        pusha                           ; save all (including new EBP)

        add     eax, ebx                ; The sum of EAX and EBX
        add     eax, ecx                ; The sum of EAX, EBX, and ECX
        add     eax, edx                ; The sum of EAX, EBX, ECX, and EDX
        mov     edx, 0                  ; Initialize for remainder
        mov     ecx, 4                  ; Set the divisor
        div     ecx                     ; divide the sum to find the average

        dump_regs 1                     ; verify registers (debug)

                                        ; Place return value in EAX
        mov     [returnvalue], eax      ; save return value in memory
        popa                            ; restore all (including new EBP)
        mov     eax, [returnvalue]      ; retrieve the saves return value

        pop     ebp                     ; restore original ebp
        ret                             ; return

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/nasm/5/solutions.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
