#!/bin/bash

# bash s_compile_cpp.sh         # runs default: 40 features, 1M nodes
# bash s_compile_cpp.sh 20 5000 # runs with 20 features and 5000 nodes

echo "compiling precision example..."
# cmd="g++ precision_test.cpp -DANNOYLIB_MULTITHREADED_BUILD -o precision_test -std=c++14 -pthread"
# eval $cmd
g++ precision_test.cpp -DANNOYLIB_MULTITHREADED_BUILD -o precision_test -std=c++14 -pthread
echo "Done compiling"

# Run the binary with optional arguments if passed
if [ $# -eq 0 ]; then
    ./precision_test
else
    ./precision_test "$@"
fi
