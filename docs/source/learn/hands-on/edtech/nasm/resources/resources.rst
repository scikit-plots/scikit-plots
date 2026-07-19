************************************************
NASM Assembly Quick Reference Guide - 32-Bit
************************************************

.. include:: ../../includes/prolog.inc

.. toctree::
   :maxdepth: 3

.. _nasm_references:

``Sources``

  - https://www.cs.uaf.edu/2005/fall/cs301/support/x86/nasm.html
  - http://www.cs.lmu.edu/~ray/notes/nasmtutorial/
  - https://www.cs.virginia.edu/~evans/cs216/guides/x86.html


Registers (Quick list:)
=============================================

.. list-table:: 7 Essential Registers
   :header-rows: 1

   * - Registers
     - Usage
   * - ``EAX ECX EDX``
     - Use and overwrite
   * - ``EBX ESP EBP ESI EDI``
     - Preserved. You must preserve the values and return them |br|
       to the registers when your program exits

Instructions
==============

.. list-table:: Quick List (useful commands)
   :widths: 25 75
   :header-rows: 1

   * - Command
     - Direction
   * - mov x, y
     - x ← y
   * - and x, y
     - x ← x and y
   * - or x, y
     - x ← x or y
   * - xor x, y
     - x ← x xor y
   * - add x, y
     - x ← x + y
   * - sub x, y
     - x ← x - y
   * - inc x
     - x ← x + 1
   * - dec x
     - x ← x - 1
   * - syscall
     - Invoke an operating system routine

.. note::
    Take note of the specific registers in the table. The operation
    stores the result in register EAX, but uses other registers.

.. list-table:: Quick List (Multiplication and Division)
   :header-rows: 1

   * - Command
     - Stored Result
     - Notes
   * - ``mul eax``
     - eax ← eax * eax
     - Results go to EAX.
   * - ``mul x``
     - eax ← eax * x
     - Multiplies EAX with the value in the other register.
   * - ``div ecx``
     - eax ← eax / ecx |br| edx ← remainder
     - Divides EAX by the value in the register by EAX. |br|
       You must clear EDX before the operation.


Detailed List
---------------

+------------------+--------------------------+-------------------------------+
| Mnemonic         | Purpose                  | Examples                      |
+------------------+--------------------------+-------------------------------+
| ``mov dest, src``| -  Move data between     | ::                            |
|                  |    registers,            |                               |
|                  | -  load immediate data   |     ; Load constant into eax  |
|                  |    into registers,       |     mov eax, 4                |
|                  | -  move data between     |     ; Copy eax into ebx       |
|                  |    registers and memory. |     mov ebx, eax              |
|                  |                          |     ; Copy ebx to memory      |
|                  |                          |     ; address 123             |
|                  |                          |     mov ebx, [123]            |
|                  |                          |                               |
+------------------+--------------------------+-------------------------------+
| ``push src``     | -  Insert a value onto   | ``push ebp``                  |
|                  |    the stack.            |                               |
|                  | -  Useful for passing    |                               |
|                  |    arguments, saving     |                               |
|                  |    registers, etc.       |                               |
+------------------+--------------------------+-------------------------------+
| ``pop dest``     | Remove topmost value     | ``pop ebp``                   |
|                  | from the stack.          |                               |
|                  | Equivalent to            |                               |
|                  |                          |                               |
|                  | -  mov dest,[esp]        |                               |
|                  | -  add esp,4             |                               |
+------------------+--------------------------+-------------------------------+
| ::               | -  Pop stack one time    | ::                            |
|                  | -  Pop stack two time    |                               |
|     esp, 4       | -  Pop stack three times |     ; Pop the last value      |
|     esp, 8       |                          |     ;from the stack           |
|     esp, 12      |                          |     esp, 4                    |
+------------------+--------------------------+-------------------------------+
| ``call func``    | -  Push the address of   | ``call print_int``            |
|                  |    the next instruction  |                               |
|                  |    |br| and start        |                               |
|                  |    executing func.       |                               |
+------------------+--------------------------+-------------------------------+
| ``ret``          | -  Pop the return        | ``ret``                       |
|                  |    program counter, and  |                               |
|                  |    jump there.           |                               |
|                  | -  Ends a subroutine.    |                               |
+------------------+--------------------------+-------------------------------+
| ``add src, dest``| -  src = src + dest      | ::                            |
|                  |                          |                               |
|                  |                          |     ; Add ebx to eax          |
|                  |                          |     add eax,ebx               |
+------------------+--------------------------+-------------------------------+
| ``mul src``      | -  Multiply eax and src  | ::                            |
|                  |    as unsigned |br|      |                               |
|                  |    integers, and put     |     ;Multiply eax by ebx      |
|                  |    the result in eax.    |     mul ebx                   |
|                  | -  High 32 bits of       |                               |
|                  |    product go into eax.  |                               |
+------------------+--------------------------+-------------------------------+
| ``jmp label``    | -  Goto the instruction  | ::                            |
|                  |    label:.               |                               |
|                  | -  Skips anything else   |     jmp post_mem              |
|                  |    in the way.           |      ...                      |
|                  |                          |     post_mem:                 |
+------------------+--------------------------+-------------------------------+
| ``cmp a,b``      | -  Compare two values.   | ``cmp eax, 10``               |
|                  | -  Sets flags that are   |                               |
|                  |    used by the           |                               |
|                  |    conditional jumps     |                               |
|                  |    (below).              |                               |
+------------------+--------------------------+-------------------------------+
| ::               | Goto label if previous   | ::                            |
|                  | comparison was:          |                               |
|     jl label     |                          |     cmp eax, 10               |
|     jle label    | -  jl <                  |                               |
|     je label     | -  jle <=                |     ; Jump if eax < 10        |
|     jge label    | -  je =                  |     jl loop_start             |
|     jg label     | -  jge >=                |                               |
|     jne label    | -  jg >                  |                               |
|     jz label     | -  jne !=                |                               |
|     jnz label    | -  jz == 0               |                               |
|                  | -  jnz !=  0             |                               |
+------------------+--------------------------+-------------------------------+
| ::               | Move if the previous     | ::                            |
|                  | comparison was:          |                               |
|     cmovl a, b   |                          |     cmp eax, 10               |
|     cmovg a, b   | -  cmov1 <               |                               |
|                  | -  cmovg >               |     ; eax = ecx  if eax  < 10 |
|                  |                          |     cmovl eax, ecx            |
+------------------+--------------------------+-------------------------------+


Registers 8-32 bit (Full List)
=================================
**Full list** of ordinary integer x86 registers.

  - **Scratch** registers any function is allowed to overwrite and use for
    anything you want without asking anybody.
  - **Preserved** registers have to be put back ("save" the register) if
    you use them.

.. list-table:: 32-bit Register List
   :header-rows: 1

   * - Notes
     - Type
     - 32-bit int
     - 16-bit short
     - 8-bit char
   * - Values are returned from |br|
       functions in this register.
     - scratch
     - eax
     - ax
     - ah and al
   * - Typical scratch register. Some |br|
       instructions also use it as a |br|
       counter.
     - scratch
     - ecx
     - cx
     - ch and cl
   * - Scratch register.
     - scratch
     - edx
     - dx
     - dh and dl
   * - Preserved register: don't use |br|
       it without saving it!
     - preserved
     - ebx
     - bx
     - bh and bl
   * - The stack pointer. Points to |br|
       the top of the stack
     - preserved
     - esp
     - sp
     -
   * - Preserved register. Sometimes |br|
       used to store the old value |br|
       of the stack pointer, or the |br|
       “base”.
     - preserved
     - ebp
     - bp
     -
   * - Preserved register. You can |br|
       use it, but you need to save |br|
       and restore it.
     - scratch
     - esi
     - si
     -
   * - Preserved register. You can |br|
       use it, but you need to save |br|
       and restore it.
     - scratch
     - edi
     - di
     -

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/nasm/resources/resources.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
