#### C++ Deep Learning Library

# The DamkeNET

The DamkeNet is a Deep Learning library that specializes in sign language recognition and was developed by Damir Bojadžić for the purposes of the “Deep Learning from Scratch in C++” practical course held in SS19 at the Technical University of Munich. The library can also be expanded and used for other DL projects.

For a detailed guide on how to use the library, please consult Documentation.pdf.
 

How to run:

1. mkdir build
2. cd build
3. cmake ..
4. make
5. copy the directory called "signlang" into build
6. copy the directory called "data" into build
7. execute src/DamkeNet


How to run tests:

Inside the src/CMakeLists.txt, in the function add_executable, replace "main.cpp" with "../tests/tests.cpp". Then execute the following commands:

1. mkdir build
2. cd build
3. cmake ..
4. make
5. mkdir datatesting
6. execute src/DamkeNet


Should you run into unexpected errors while compiling, please make sure that all neccessary header files were included and contact the author immediately.
