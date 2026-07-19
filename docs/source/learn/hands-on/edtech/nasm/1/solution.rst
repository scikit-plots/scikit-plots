*****************************
Lab 1: Solution Files
*****************************

hello.asm
==========

.. code-block:: text

    ; include directives
    global asm_main
    %include "../lib/asm_io.inc"

    segment .data
        ; DX directives (read only, static data)
        hello    db    "Hello World! Привет, мир! Сәлем Әлем!¡Hola Mundo!", 0xA, 0
        world    db    "¡Hola Mundo!", 0xA, "नमस्ते दुनिया!", 0xA, "안녕하세요!", 0xA,  "你好", 0xA, "世界!", 0xA, 0

    segment .bss
        ; RESX directives (writable data)

    segment .text
    asm_main:                    ; entry point (called from driver.c)
        ; instructions

        enter 0,0                ; Initialize register EBP
        pusha                    ; Push register contents to the stack
        ;
        mov     eax,  hello      ; put hello in EAX
        call    print_string     ; print data in EAX
        mov     eax,  world      ; put world in EAX
        call    print_string     ; print data in EAX
        ;
        popa                     ; Restore register data from the stack
        mov eax, 0               ; flush EAX of data
        leave                    ; restore the stack
        ret                      ; Return from main back into C library wrapper


Makefile
===========

.. code-block:: make

    # compiles the project files (hello.asm)
    # Compiles dependencies if they are not up to date
    hello: asm_io.o driver.o hello.o
        @echo "Building hello.asm"
        @gcc -m32 driver.o hello.o asm_io.o -o hello
        @rm hello.o		# Remove hello.o to force a recompile each time
        ./hello

    # Build hello.asm
    hello.o:
        @nasm -f elf hello.asm -o hello.o

    # Build asm_io.asm
    asm_io.o:
        @nasm -f elf -d ELF_TYPE ../lib/asm_io.asm -o asm_io.o

    # Build driver.c
    driver.o:
        @gcc -m32 -c ../lib/driver.c -o driver.o

    # Clean up and then rebuild all (asm_io.asm, driver.c, and hello.asm)
    all: clean hello

    # Removes all .o and temporary files
    clean:
        @echo "Cleaning up..."
        @rm -f *.o
        @rm -f hello

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/nasm/1/solution.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
