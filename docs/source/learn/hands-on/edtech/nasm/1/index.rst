*****************************
Lab 1: Building an ASM File
*****************************

.. note::
  * These instructions are written for POSIX (Ubuntu Linux 18.04.3, x86_64).
  * The libraries from |PC Assembly| are 32-bit only

.. include:: ../../includes/prolog.inc

.. include:: ../asm-urls.rst

.. contents:: Table of Contents

Overview
==========
You will write a Hello World program using assembly code! Also, you will
learn how to create a ``Makefile`` to build a project with multiple
includes or dependencies.

Makefile
-------------


NASM
-------

.. _posix_env_setup:

Task 1: POSIX Environment Setup
================================================
Firstly, let's set up the environment to compile and link assembly code.

.. note::
    We need to install the development tools for 32-bit because
    the ``asm_io.inc`` library works for 32-bit only.

#. Install core dev 32bit GCC tools

   .. code-block:: bash

       sudo apt update
       sudo apt install build-essential nasm gcc-multilib g++-multilib

#. Verify that essential tools are installed

   .. code-block:: bash

       gcc --version
       nasm --version
       make --version

#. Download the sample files libraries from |PC Assembly|

   .. code-block:: bash

       # Create your project directory
       mkdir ~/asm
       cd ~/asm

       # Download and extract the .zip file
       wget https://pacman128.github.io/static/linux-ex.zip
       unzip linux-ex.zip
       mv linux-ex lib

#. Verify that your files are in ``lib`` dir (or the folder of your choice)

   .. code-block:: bash

       ls -lh ~/asm/lib

Task 2: Create a ``HelloWorld`` Assembly Program
===================================================

#. Create a project folder and then change to that folder

   .. code-block:: bash

      mkdir ~/asm/hello
      cd ~/asm/hello
      touch hello.asm

#. Edit the file using ``nano hello.asm`` or using a graphical editor

   .. code-block:: asm
       :linenos:

       ; include directives
       global asm_main

       segment .data
           ; DX directives (read only, static data)

       segment .bss
           ; RESX directives (writable data)

       segment .text
           asm_main:
           ; instructions

#. Add this code to the the ``.text`` section. This code initializes
   the process and then cleans up afterwards.

   .. code-block:: asm
       :linenos:

       enter 0,0        ; Initialize register EBP
       pusha            ; Push register contents to the stack
       ;
       ; Your program here
       ;
       popa             ; Restore register data from the stack
       mov eax, 0       ; flush EAX of data
       leave            ; restore the stack
       ret              ; Return from main back into the C library wrapper

   **Here is a brief explanation of what this |enter and leave code| does:**

   enter 0,0
     |enter 0,0| pushes register EBP to the stack to preserve the value and then
     initialize the register for use.

   pusha
     |pusha| pushes the contents of the general-purpose registers onto the stack

   enter, leave
     |enter, leave| set up the stack for use. They are shorthand
     instructions that are equivalent to these commands:

     .. code-block:: asm
         :caption: enter and leave are like macros for

         ; enter
         push ebp
         mov ebp, esp

         ; leave
         mov esp, ebp
         pop ebp

#. Add the library to the include directives section.

   ::

       %include "../lib/asm_io.inc"

#. Add the ``hello world`` text to the DX directive section.

   ::

       hello    db    "Hello, world!", 0xA, 0

   .. note:: Think: What is the purpose of ``0xA`` and ``0``?
       Why are they there?

#. Copy the data from ``hello`` to ``eax`` and then call ``print_string``

   .. important:: ``print`` will display the value in register ``eax``

   .. code-block:: asm

       mov     eax,  hello    ; put hello in EAX
       call    print_string   ; print data in EAX

The completed file should look like this:

    .. code-block:: text
        :caption: hello.asm

        ; include directives
        global asm_main
        %include "../lib/asm_io.inc"

        segment .data
            ; DX directives (read only, static data)
            hello    db    "Hello, world!", 0xA, 0

        segment .bss
            ; RESX directives (writable data)

        segment .text
        asm_main:                   ; entry point (called from driver.c)
            ; instructions

            enter 0,0               ; Initialize register EBP
            pusha                   ; Push register contents to the stack
            ;

            mov     eax,  hello     ; put hello in EAX
            call    print_string    ; print data in EAX

            ;
            popa                    ; Restore register data from the stack
            mov eax, 0              ; flush EAX of data
            leave                   ; restore the stack
            ret                     ; Return from main back into the C library wrapper

Task 3: Build the project
============================

Build Overview
--------------

The build process uses multiple steps. See the graphic in
|x86 Assembly Language Programming|.

  #. First, you create the object files (.o files) using NASM or GCC.
     This procedure compiles to ELF (Executable and Linkable Format)
  #. Then, you link the object files using a linker to create a file that
     contains machine code (or an executable).

**Additional Resources**

  - |Nasihatkon - Lecture3.pdf|
  - |First NASM Program (module)| - University of Hawaii
  - |NASM first program| - University of Hawaii Presentation

You will compile these files:

  asm_io.asm
    Contains labels (like macros or functions) that perform common IO
    operations, such as ``print_string`` or ``print_int``.

  driver.c
    Executes the code in the ``.asm`` file by calling ``asm_main()``

    .. code-block:: c
        :emphasize-lines: 4

        int main()
        {
            int ret_status;
            ret_status = asm_main();
            return ret_status;
        }

    .. note::
        ``driver.c`` is not required if you change ``global asm_main`` to
        ``global main`` in the .asm file.

        C provides a convenient wrapper for our ASM files. We will need to use
        this method on later projects when we integrate C and ASM.

The Build Process
------------------
#. First, we need to build ``asm_io.asm`` and ``driver.c``. **We need to
   do this once**.

   .. note:: Pay attention to the relative paths. ``../`` or `./`

   .. code-block:: bash

       # First, build asm_io.asm and driver.c
       # The output is a 32-bit ELF (Executable and Linkable Format) file
       nasm -f elf -d ELF_TYPE ../lib/asm_io.asm -o asm_io.o
       gcc -m32 -c ../lib/driver.c -o driver.o

#. Then, build ``hello.asm``.

   .. code-block:: bash

       # Next, build the project file to ELF
       nasm -f elf hello.asm -o hello.o

#. Lastly, link the ``.o`` files to create a single executable that
   contains the machine code for the target CPU. You can then
   execute the file.

   .. code-block:: bash

       # Lastly, link files to create ``hello``
       gcc -m32 driver.o hello.o asm_io.o -o hello
       ./hello

The entire process looks like this:

.. code-block:: bash
    :linenos:

    # To compile to ELF (Executable and Linkable Format)

    # First, build asm_io.asm and driver.c
    nasm -f elf -d ELF_TYPE ../lib/asm_io.asm -o asm_io.o
    gcc -m32 -c ../lib/driver.c -o driver.o

    # Then, build and link the file
    nasm -f elf hello.asm -o hello.o
    gcc -m32 driver.o hello.o asm_io.o -o hello

    # Execute the file
    ./hello

Task 4: Create a simple Makefile
===================================
A ``Makefile`` contains commands to simplify the build process. We don't
want to manually repeat the process in the last manually. We can
create different rules to recompile all dependencies or just our project
files. We can also include a rule to build for specific platforms that
might required additional libraries.

Follow the *basic examples* from |What is a Makefile and how does it work?|
to learn how to write a Makefile to automate a build process.

Here are some resources to help you.

  - https://devhints.io/makefile
  - http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
  - https://www.cs.swarthmore.edu/~newhall/unixhelp/howto_makefiles.html
  - https://www.gnu.org/software/make/manual/html_node/Simple-Makefile.html

.. note::
    - A Makefile requires the use of tabs (ASCII symbol ``HT``,
      code ``0x09``) instead of blank spaces.
    - HTML does not display a tab or HT. You will need to copy/paste
      a HT from notepad if your text editor does not support tabs.
    - Here is the error code that you will receive if you do not use
      a tab on line number 2.

      .. code-block:: bash

          Makefile:2: *** missing separator.  Stop.

Task 5: Create a Makefile for hello.asm
=========================================
#. Create a Makefile to build the project using these targets in the template.
   file. We've completed target ``hello`` for you. Fill in the others.

     .. code-block:: make
         :caption: Makefile targets for hello.asm

         # - Creates a single executable (hello)
         # - compiles dependencies (asm_io.o driver.o hello.asm) if they are not up to date
         hello: asm_io.o driver.o hello.o
             @gcc -m32 asm_io.o driver.o hello.o -o hello
             @rm hello.o           # Remove hello.o to force a recompile each time
             ./hello               # Execute hello

         # Compile hello.asm
         hello.o:


         # Compile asm_io.asm
         asm_io.o:


         # Compile driver.c
         driver.o:


         # Clean up and then rebuild all (asm_io.asm, driver.c, and hello.asm)
         all: clean hello

         # Removes all build and temporary files
         clean:
             @echo "Cleaning up..."

   Here is the expected output for each command.

   make clean
       ``make clean`` removes all non-source files.

       .. code-block:: bash
           :caption: make clean
           :emphasize-lines: 1,9,11

               sysadmin@ubuntu:~/asm/hello$ ls -lh
               total 32K
               -rw-rw-r-- 1 sysadmin sysadmin 3.6K Feb 28 15:28 asm_io.o
               -rw-rw-r-- 1 sysadmin sysadmin 1.3K Feb 28 15:28 driver.o
               -rwxrwxr-x 1 sysadmin sysadmin 9.2K Feb 28 15:28 hello
               -rw-rw-r-- 1 sysadmin sysadmin  719 Feb 28 13:58 hello.asm
               -rw-rw-r-- 1 sysadmin sysadmin  928 Feb 28 15:28 hello.o
               -rw-rw-r-- 1 sysadmin sysadmin  580 Feb 28 15:28 Makefile
               sysadmin@ubuntu:~/asm/hello$ make clean
               Cleaning up...
               sysadmin@ubuntu:~/asm/hello$ ls -lh
               total 8.0K
               -rw-rw-r-- 1 sysadmin sysadmin 719 Feb 28 13:58 hello.asm
               -rw-rw-r-- 1 sysadmin sysadmin 580 Feb 28 15:28 Makefile
               sysadmin@ubuntu:~/asm/hello$

   make
       ``make`` calls the default target of ``hello``. This target will build
       the project if the files do not exist or if they are out of date.

       Notice the contents of the directory and that the time stamps of the
       dependent files do not change.

       .. code-block:: bash
           :caption: make
           :emphasize-lines: 1,7,13-14,25-26

           sysadmin@ubuntu:~/asm/hello$ make clean
           Cleaning up...
           sysadmin@ubuntu:~/asm/hello$ ls -lh
           total 8.0K
           -rw-rw-r-- 1 sysadmin sysadmin 719 Feb 28 13:58 hello.asm
           -rw-rw-r-- 1 sysadmin sysadmin 574 Feb 28 15:10 Makefile
           sysadmin@ubuntu:~/asm/hello$ make
           Building hello.asm
           ./hello
           Hello, world!
           sysadmin@ubuntu:~/asm/hello$ ls -lh
           total 32K
           -rw-rw-r-- 1 sysadmin sysadmin 3.6K Feb 28 15:10 asm_io.o
           -rw-rw-r-- 1 sysadmin sysadmin 1.3K Feb 28 15:10 driver.o
           -rwxrwxr-x 1 sysadmin sysadmin 9.2K Feb 28 15:10 hello
           -rw-rw-r-- 1 sysadmin sysadmin  719 Feb 28 13:58 hello.asm
           -rw-rw-r-- 1 sysadmin sysadmin  928 Feb 28 15:10 hello.o
           -rw-rw-r-- 1 sysadmin sysadmin  574 Feb 28 15:10 Makefile
           sysadmin@ubuntu:~/asm/hello$ make
           Building hello.asm
           ./hello
           Hello, world!
           sysadmin@ubuntu:~/asm/hello$ ls -lh
           total 32K
           -rw-rw-r-- 1 sysadmin sysadmin 3.6K Feb 28 15:10 asm_io.o
           -rw-rw-r-- 1 sysadmin sysadmin 1.3K Feb 28 15:10 driver.o
           -rwxrwxr-x 1 sysadmin sysadmin 9.2K Feb 28 15:15 hello
           -rw-rw-r-- 1 sysadmin sysadmin  719 Feb 28 13:58 hello.asm
           -rw-rw-r-- 1 sysadmin sysadmin  928 Feb 28 15:15 hello.o
           -rw-rw-r-- 1 sysadmin sysadmin  574 Feb 28 15:10 Makefile
           sysadmin@ubuntu:~/asm/hello$


   make asm_io.o (driver.o, hello.o)
       ``make asm_io.o``, ``make driver.o``, and ``make hello.o``, will compile
       the .asm file to .o. The file will not recompile if it is up to date.

       .. code-block:: bash
           :caption: make file.o
           :emphasize-lines: 3,5

           sysadmin@ubuntu:~/asm/hello$ make clean
           Cleaning up...
           sysadmin@ubuntu:~/asm/hello$ make asm_io.o
           sysadmin@ubuntu:~/asm/hello$ make asm_io.o
           make: 'asm_io.o' is up to date.
           sysadmin@ubuntu:~/asm/hello$

   make all
       ``make driver.all`` does a complete rebuild. It deletes all build
       files and then recompiles each file. The target calls two other targets,
       ``clean`` and ``hello``

       .. code-block:: bash
           :caption: make all
           :emphasize-lines: 1

           sysadmin@ubuntu:~/asm/hello$ make all
           Cleaning up...
           Building hello.asm
           ./hello
           Hello, world!
           sysadmin@ubuntu:~/asm/hello$

#. Modify ``hello.asm`` by changing the hello world text to:

   .. code-block:: text

       "Hello World!! :D! Привет, мир! Сәлем Әлем!"

#. Execute ``make`` to verify that your project rebuild successfully.

   .. code-block:: bash

       sysadmin@ubuntu:~/asm/hello$ nano hello.asm
       sysadmin@ubuntu:~/asm/hello$
       sysadmin@ubuntu:~/asm/hello$ make
       Building hello.asm
       ./hello
       Hello World! Привет, мир! ¡Hola Mundo!
       नमस्ते दुनिय! 你好，世界!
       sysadmin@ubuntu:~/asm/hello$

#. Add a second ``label`` and ``string``.

   .. code-block:: text

       world    db    "¡Hola Mundo!", 0xA, "नमस्ते दुनिया!", 0xA, "안녕하세요!", 0xA,  "你好", 0xA, "世界!", 0xA, 0

#. Put the new string in ``eax`` and then call ``print_string``

   .. code-block:: asm

       mov     eax,  world   ; put world in EAX
       call    print_string   ; print data in EAX

#. Make the project to see the new output.

   .. code-block:: bash

       sysadmin@ubuntu:~/asm/hello$ make
       Building hello.asm
       ./hello
       Hello World! Привет, мир! Сәлем Әлем!
       ¡Hola Mundo!
       नमस्ते दुनिया!
       안녕하세요!
       你好
       世界!

See the `next page <solution.html>`_ for the solutions.

.. toctree::
    :hidden:

    solution

.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/nasm/1/index.html

   See :doc:`LICENSE <../../LICENSE_edtech>` for the full license text.
