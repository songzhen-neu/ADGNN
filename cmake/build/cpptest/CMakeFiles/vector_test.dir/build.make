# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/songzhen/workspace/ADGNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/songzhen/workspace/ADGNN/cmake/build

# Include any dependencies generated for this target.
include cpptest/CMakeFiles/vector_test.dir/depend.make

# Include the progress variables for this target.
include cpptest/CMakeFiles/vector_test.dir/progress.make

# Include the compile flags for this target's objects.
include cpptest/CMakeFiles/vector_test.dir/flags.make

cpptest/CMakeFiles/vector_test.dir/vector_test.cc.o: cpptest/CMakeFiles/vector_test.dir/flags.make
cpptest/CMakeFiles/vector_test.dir/vector_test.cc.o: ../../cpptest/vector_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cpptest/CMakeFiles/vector_test.dir/vector_test.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vector_test.dir/vector_test.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/vector_test.cc

cpptest/CMakeFiles/vector_test.dir/vector_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_test.dir/vector_test.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/vector_test.cc > CMakeFiles/vector_test.dir/vector_test.cc.i

cpptest/CMakeFiles/vector_test.dir/vector_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_test.dir/vector_test.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/vector_test.cc -o CMakeFiles/vector_test.dir/vector_test.cc.s

# Object files for target vector_test
vector_test_OBJECTS = \
"CMakeFiles/vector_test.dir/vector_test.cc.o"

# External object files for target vector_test
vector_test_EXTERNAL_OBJECTS =

cpptest/vector_test: cpptest/CMakeFiles/vector_test.dir/vector_test.cc.o
cpptest/vector_test: cpptest/CMakeFiles/vector_test.dir/build.make
cpptest/vector_test: cpptest/CMakeFiles/vector_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable vector_test"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vector_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpptest/CMakeFiles/vector_test.dir/build: cpptest/vector_test

.PHONY : cpptest/CMakeFiles/vector_test.dir/build

cpptest/CMakeFiles/vector_test.dir/clean:
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && $(CMAKE_COMMAND) -P CMakeFiles/vector_test.dir/cmake_clean.cmake
.PHONY : cpptest/CMakeFiles/vector_test.dir/clean

cpptest/CMakeFiles/vector_test.dir/depend:
	cd /home/songzhen/workspace/ADGNN/cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN/cpptest /home/songzhen/workspace/ADGNN/cmake/build /home/songzhen/workspace/ADGNN/cmake/build/cpptest /home/songzhen/workspace/ADGNN/cmake/build/cpptest/CMakeFiles/vector_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpptest/CMakeFiles/vector_test.dir/depend

