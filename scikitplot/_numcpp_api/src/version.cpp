// src/version.cpp
// C++ Source File
#include <iostream>
#include <string>

// nc::VERSION
#include "Version.hpp"  // Include the Version.hpp file

// extern "C" disables name mangling, making the function callable from C
// and other languages that expect C-style linkage.
// Without extern "C", name mangling is enabled,
// and the function is only directly usable from C++.
// C++ function declaration with a default message
extern "C" const char* get_version() {
  // Print the version
  // std::cout << nc::VERSION << std::endl;
  // Return the version as a const char* from the nc::VERSION array
  return nc::VERSION;  // Return nc::VERSION is a const char[], no need to call c_str()
}