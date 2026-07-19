**********************************
Lab 2: User Input and Arithmetic
**********************************

.. include:: ../../includes/prolog.inc

.. include:: ../asm-urls.rst

.. contents:: Table of Contents

Overview
==========
Your company is programming an embedded device that can manually adjust
the settings of laboratory equipment. The technicians will type in a value
using a terminal and then receive an output depending on how the machine
responds.

You will learn about user input and basic calculations in NASM.

Sources
----------
- |Introduction to NASM Programming| - University of Hawaii Presentation
- |NASM: data and bss (inverted)| - University of Hawaii Presentation
- |x86 Assembly Language Programming| - Ray Toal
- |X86 Assembly/Arithmetic| - Wikibooks

Registers and Pointers
------------------------

Recall in C that you can work with the value directly or work with
a value at a specific memory location.

  - ``&`` gets the address of the variable.
  - ``*`` get the value of the variable at the memory location.

  Consider this code block:

  .. code-block:: c

      int x = 5;              // Assign x to a value
      int *address = &x;      // Get the address of x

      printf("Value of 'x':       = %d; \n", x);
      printf("Address of 'x'      = %p \n", address);
      printf("Contents at address = %d; ", *address);

      /*
          Output:
          Value of 'x':       = 5;
          Address of 'x'      = 0x7ffc62289f14
          Contents at address = 5;
      */

ASM operates in a similar way. See slides 22 and 24 in
|Introduction to NASM Programming|.

    .. code-block:: asm

        mov  eax, 5      ; places the value of 5 in register EAX
                         ; eax = 5

        mov  ecx, [eax]  ; places the contents at the address of EAX in ECX
                         ; ecx = *eax

Reserving Memory
------------------
The programmer is responsible for reserving to prevent a
|segmentation fault| and other errors. There are two types of data
directives in an ASM file to reserve memory, which ``DX`` and ``RESX``.

The ``X`` refers to the data size.

.. csv-table:: DX/RESX Data Unit
    :header: "Unit", "Letter (X)", "Size in bytes"
    :widths: 30, 10, 10

    byte,b,1
    word,w,2
    double word,d,4
    quad word,q,8
    ten bytes,t,10

**DX directives: Initialized Data (D = "defined")**

Memory is reserved in the DX block using the three elements:

  #. label - An unquoted text string
  #. ``d`` + memory unit
  #. initial value

.. code-block:: text
    :caption: DX Examples

    L1    db    0x55                ; just the byte 0x55
    L2    db    0x55,0x56,0x57      ; three bytes in succession
    L3    db    'a',0x55            ; character constants are OK
    L4    db    'hello',13,10,'$'   ; so are string constants
    L5    dw    0x1234              ; 0x34 0x12
    L6    dw    'a'                 ; 0x61 0x00 (it's just a number)
    L7    dw    'ab'                ; 0x61 0x62 (character constant)
    L8    dw    'abc'               ; 0x61 0x62 0x63 0x00 (string)
    L9    dd    0x12345678          ; 0x78 0x56 0x34 0x12
    L10   dd    1.234567e20         ; floating-point constant
    L11   dq    0x123456789abcdef0  ; eight byte constant
    L12   dq    1.234567e20         ; double-precision float
    L13   dt    1.234567e20         ; extended-precision float

**RESX directives: Uninitialized Data (RES = "reserved")**

Memory is reserved in the RESX block using the three elements:

  #. label - An unquoted text string
  #. ``res`` + memory unit
  #. memory to reserve

.. code-block:: asm
    :caption: RESX Examples

    buffer:       resb    64        ; reserve 64 bytes
    wordvar:      resw    1         ; reserve a word
    realarray:    resq    10        ; array of ten reals (floats)


Task 1: Setup your Project
================================================

#. Create your project directory, ``input.asm`` file, and ``Makefile``.
#. Use :ref:`nasm_template` to help you get started.

.. tip::
    Create a template folder that you can replicate for each project.

    .. code-block:: bash

        # Create a copy for a new project
        cd ~/asm
        cp -r template lab2


Task 2: Prompt for Input and Print the values
===================================================
You will verify that your program can read in a value and display it.

Use |Introduction to NASM Programming| and |NASM: data and bss (inverted)|
to help you.

#. Display a message to the user to input a value. The string
   to print goes in the ``.data`` segment.
#. Read the input using ``read_int``, which get put in register EAX.

   .. code-block:: asm

       call    read_int     ; read_int puts the value in EAX

#. Print the value using ``print_int``

   .. code-block:: text
       :caption: Expected Output

       Enter a value: 358
       358

   .. note::
       The value resides in EAX, but we will lose the value if we
       don't store it in memory.

#. Create a label for the input value in the ``.bss`` segment and reserve
   memory for a 32-bit value, which is 4 bytes or a ``double word``.

   - Call the label ``VAL1``
   - Choose a method to reserve 32 bits

   .. code-block:: asm

       VAL1    resd    1  ; reserve a single double word
       VAL1    resb    4  ; reserve 4 bytes

#. Store the value in EAX in memory using the address at VAL1.

   .. note::
       This operation works with the memory address, similar to how C
       uses a pointer with ``scanf``.

       .. code-block:: C

           char *input;          // Get pointer to a single byte
           scanf("%s",&input);   // Read a single byte and store in memory

       You would do this in ASM:

       .. code-block:: asm

           mov    [VAL1], eax

#. Prompt the user for a second value and store it in memory using a
   label called ``VAL2``.
#. Print the first value that label ``VAL1`` references. You must first
   move the value to EAX.

   .. code-block:: asm

       mov    eax, [VAL1]

#. Verify that you can input and output the full range of signed
   32-bit values. Then, try entering more data than can fit in
   a 32-bit register (EAX).

.. code-block:: text
    :caption: Expected Output

    ; Minimum range
    Enter a value: 1
    Enter a value: -1
    1
    -1

    ; Maximum range
    Enter a value: 2147483647
    Enter a value: -2147483648
    2147483647
    -2147483648

    ; Values too large for the register are truncated
    Enter a value: 9999999999
    Enter a value: -9999999999
    2147483647
    -2147483648

Task 3: Addition an Subtraction
=================================
Your program now accepts user input, stores the value
in memory, and then prints the value to the screen.

It more interesting to do something with the data instead of
reprinting the input.

.. tip::
    - Move the data to a register to manipulate it.
    - Use registers ``EAX``, ``ECX``, and ``EDX`` for general use or
      scratch registers.

Here is a quick list of use NASM commands for manipulating data.


.. list-table:: Quick List (useful commands)
   :widths: 25 75
   :header-rows: 1

   * - Command
     - Stored Result
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
     - x ← x – y
   * - inc x
     - x ← x + 1
   * - dec x
     - x ← x – 1

#. Move the value in ``VAL1`` to ECX.
#. Move the value in ``VAL2`` to EDX.
#. Increment the first input placed in ECX.
#. Decrement the second input placed in EDX.
#. Print the results.

   .. note:: You will have to move the values to EAX to print them

   .. code-block:: text
       :caption: Expected Results (comments added for clarity)

       Enter a value: 10
       Enter a value: 20
       10       ; [VAL1]
       20       ; [VAL2]

       11       ; [VAL1]++
       19       ; [VAL2]--

#. Add the two values together and store the value in ``RESX`` with a
   label of ``RESULT``.
#. Print the value in ``RESULT``.

   .. code-block:: text
       :caption: Expected Results

       Enter a value: 10
       Enter a value: 20
       10       ; [VAL1]
       20       ; [VAL2]

       11       ; [VAL1]++
       19       ; [VAL2]--

       30       ; [RESULT] = [VAL1] + [VAL2]

#. Subtract ``VAL1`` or ``VAL2`` from ``RESULT`` and verify that value
   matches the original. You can call ``dump_regs 0`` to view the contents
   of the registers.

   .. code-block:: text
       :caption: Expected Results

       Enter a value: 10
       Enter a value: 20
       10       ; [VAL1]
       20       ; [VAL2]

       11       ; [VAL1]++
       19       ; [VAL2]--

       30       ; [RESULT] = [VAL1] + [VAL2]

       11       ; [RESULT] = [RESULT] - [VAL2]

       Register Dump # 0
       EAX = 0000000B EBX = 56649FCC ECX = 0000000B EDX = 00000013
       ESI = F7FAF000 EDI = 00000000 EBP = FFD870C8 ESP = FFD870A8
       EIP = 56648851 FLAGS = 0202

Task 4: Multiplication and Division
====================================
You should have a good feel for working with registers and RESX
directives. Now, we can do more complex operations by performing
multiplication and division.

.. note::
    - Take note of the specific registers in the table.
    - The operations stores the result in register ``EAX``, but use
      register ``EDX`` to store additional data.

.. list-table:: Quick List (Multiplication and Division)
   :header-rows: 1

   * - Command
     - Stored Result
     - Notes
   * - ``mul eax``
     - eax ← eax * eax |br| edx ← higher
     -
       - Lower results go in EAX.
       - Higher results go in EDX.
   * - ``mul x``
     - eax ← eax * x
     -
       - Multiplies EAX with the value in the other register.
       - Uses register EDX if the Carry Flag is set.
       - See |X86 Assembly/Arithmetic| and |br|
         |Multiplication and Division Instructions|.

   * - ``div ecx``
     - eax ← eax / ecx |br| edx ← remainder
     - Divides EAX by the value in the register by EAX. |br|
       You must clear EDX before the operation.

.. note::
    You should clear register EDX before performing a multiplication
    or division operation.

#. Multiply the value in EAX by the value in ``VAL1`` and print the results.

   **Note**: You need to put ``VAL1`` in register ECX.

   .. code-block:: text
       :caption: Expected Results

       Enter a value: 10
       Enter a value: 20
       10       ; [VAL1]
       20       ; [VAL2]

       11       ; [VAL1]++
       19       ; [VAL2]--

       30       ; [RESULT] = [VAL1] + [VAL2]

       11       ; [RESULT] = [RESULT] - [VAL2]

       110      ; [RESULT] = EAX * [VAL1]

       Register Dump # 0
       EAX = 0000006E EBX = 56580FCC ECX = 0000000A EDX = 00000000
       ESI = F7EC7000 EDI = 00000000 EBP = FFBB11E8 ESP = FFBB11C8
       EIP = 5657F878 FLAGS = 0202

#. Now, divide the value in EAX by the value in ``VAL2``. Print the result
   of the division (EAX) and of the remainder (EDX).

   ``110 / 20 = 5, remainder 10``

   .. note::
       Evaluate the results of ``dump_regs`` to see the parts equation
       and result.

   .. code-block:: text
       :caption: Expected Results

       Enter a value: 10
       Enter a value: 20
       10       ; [VAL1]
       20       ; [VAL2]

       11       ; [VAL1]++
       19       ; [VAL2] --

       30       ; [RESULT] = [VAL1] + [VAL2]

       11       ; [RESULT] = [RESULT] - [VAL2]

       110      ; EAX = EAX * [VAL1]

       5        ; EAX = EAX / [VAL2]
       10       ; Remainder in EDX

       Register Dump # 0
       EAX = 0000000A EBX = 5660DFCC ECX = 00000014 EDX = 0000000A
       ESI = F7F25000 EDI = 00000000 EBP = FF9939A8 ESP = FF993988
       EIP = 5660C89B FLAGS = 0202

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/nasm/2/index.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
