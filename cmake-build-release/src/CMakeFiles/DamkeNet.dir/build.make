# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/mrtheamir/CLion/clion-2019.1/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/mrtheamir/CLion/clion-2019.1/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mrtheamir/Desktop/libdl

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mrtheamir/Desktop/libdl/cmake-build-release

# Include any dependencies generated for this target.
include src/CMakeFiles/DamkeNet.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/DamkeNet.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/DamkeNet.dir/flags.make

src/CMakeFiles/DamkeNet.dir/main.cpp.o: src/CMakeFiles/DamkeNet.dir/flags.make
src/CMakeFiles/DamkeNet.dir/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mrtheamir/Desktop/libdl/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/DamkeNet.dir/main.cpp.o"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DamkeNet.dir/main.cpp.o -c /home/mrtheamir/Desktop/libdl/src/main.cpp

src/CMakeFiles/DamkeNet.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DamkeNet.dir/main.cpp.i"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mrtheamir/Desktop/libdl/src/main.cpp > CMakeFiles/DamkeNet.dir/main.cpp.i

src/CMakeFiles/DamkeNet.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DamkeNet.dir/main.cpp.s"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mrtheamir/Desktop/libdl/src/main.cpp -o CMakeFiles/DamkeNet.dir/main.cpp.s

src/CMakeFiles/DamkeNet.dir/net.cpp.o: src/CMakeFiles/DamkeNet.dir/flags.make
src/CMakeFiles/DamkeNet.dir/net.cpp.o: ../src/net.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mrtheamir/Desktop/libdl/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/DamkeNet.dir/net.cpp.o"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DamkeNet.dir/net.cpp.o -c /home/mrtheamir/Desktop/libdl/src/net.cpp

src/CMakeFiles/DamkeNet.dir/net.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DamkeNet.dir/net.cpp.i"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mrtheamir/Desktop/libdl/src/net.cpp > CMakeFiles/DamkeNet.dir/net.cpp.i

src/CMakeFiles/DamkeNet.dir/net.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DamkeNet.dir/net.cpp.s"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mrtheamir/Desktop/libdl/src/net.cpp -o CMakeFiles/DamkeNet.dir/net.cpp.s

src/CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.o: src/CMakeFiles/DamkeNet.dir/flags.make
src/CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.o: ../tests/FeedForward.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mrtheamir/Desktop/libdl/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.o"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.o -c /home/mrtheamir/Desktop/libdl/tests/FeedForward.cpp

src/CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.i"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mrtheamir/Desktop/libdl/tests/FeedForward.cpp > CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.i

src/CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.s"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mrtheamir/Desktop/libdl/tests/FeedForward.cpp -o CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.s

src/CMakeFiles/DamkeNet.dir/Layer.cpp.o: src/CMakeFiles/DamkeNet.dir/flags.make
src/CMakeFiles/DamkeNet.dir/Layer.cpp.o: ../src/Layer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mrtheamir/Desktop/libdl/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/DamkeNet.dir/Layer.cpp.o"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DamkeNet.dir/Layer.cpp.o -c /home/mrtheamir/Desktop/libdl/src/Layer.cpp

src/CMakeFiles/DamkeNet.dir/Layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DamkeNet.dir/Layer.cpp.i"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mrtheamir/Desktop/libdl/src/Layer.cpp > CMakeFiles/DamkeNet.dir/Layer.cpp.i

src/CMakeFiles/DamkeNet.dir/Layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DamkeNet.dir/Layer.cpp.s"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mrtheamir/Desktop/libdl/src/Layer.cpp -o CMakeFiles/DamkeNet.dir/Layer.cpp.s

# Object files for target DamkeNet
DamkeNet_OBJECTS = \
"CMakeFiles/DamkeNet.dir/main.cpp.o" \
"CMakeFiles/DamkeNet.dir/net.cpp.o" \
"CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.o" \
"CMakeFiles/DamkeNet.dir/Layer.cpp.o"

# External object files for target DamkeNet
DamkeNet_EXTERNAL_OBJECTS =

src/DamkeNet: src/CMakeFiles/DamkeNet.dir/main.cpp.o
src/DamkeNet: src/CMakeFiles/DamkeNet.dir/net.cpp.o
src/DamkeNet: src/CMakeFiles/DamkeNet.dir/__/tests/FeedForward.cpp.o
src/DamkeNet: src/CMakeFiles/DamkeNet.dir/Layer.cpp.o
src/DamkeNet: src/CMakeFiles/DamkeNet.dir/build.make
src/DamkeNet: src/CMakeFiles/DamkeNet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mrtheamir/Desktop/libdl/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable DamkeNet"
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DamkeNet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/DamkeNet.dir/build: src/DamkeNet

.PHONY : src/CMakeFiles/DamkeNet.dir/build

src/CMakeFiles/DamkeNet.dir/clean:
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release/src && $(CMAKE_COMMAND) -P CMakeFiles/DamkeNet.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/DamkeNet.dir/clean

src/CMakeFiles/DamkeNet.dir/depend:
	cd /home/mrtheamir/Desktop/libdl/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mrtheamir/Desktop/libdl /home/mrtheamir/Desktop/libdl/src /home/mrtheamir/Desktop/libdl/cmake-build-release /home/mrtheamir/Desktop/libdl/cmake-build-release/src /home/mrtheamir/Desktop/libdl/cmake-build-release/src/CMakeFiles/DamkeNet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/DamkeNet.dir/depend

