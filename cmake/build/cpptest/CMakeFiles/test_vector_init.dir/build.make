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
include cpptest/CMakeFiles/test_vector_init.dir/depend.make

# Include the progress variables for this target.
include cpptest/CMakeFiles/test_vector_init.dir/progress.make

# Include the compile flags for this target's objects.
include cpptest/CMakeFiles/test_vector_init.dir/flags.make

cpptest/CMakeFiles/test_vector_init.dir/testVectorInit.cpp.o: cpptest/CMakeFiles/test_vector_init.dir/flags.make
cpptest/CMakeFiles/test_vector_init.dir/testVectorInit.cpp.o: ../../cpptest/testVectorInit.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cpptest/CMakeFiles/test_vector_init.dir/testVectorInit.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_vector_init.dir/testVectorInit.cpp.o -c /home/songzhen/workspace/ADGNN/cpptest/testVectorInit.cpp

cpptest/CMakeFiles/test_vector_init.dir/testVectorInit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_vector_init.dir/testVectorInit.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/testVectorInit.cpp > CMakeFiles/test_vector_init.dir/testVectorInit.cpp.i

cpptest/CMakeFiles/test_vector_init.dir/testVectorInit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_vector_init.dir/testVectorInit.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/testVectorInit.cpp -o CMakeFiles/test_vector_init.dir/testVectorInit.cpp.s

# Object files for target test_vector_init
test_vector_init_OBJECTS = \
"CMakeFiles/test_vector_init.dir/testVectorInit.cpp.o"

# External object files for target test_vector_init
test_vector_init_EXTERNAL_OBJECTS =

cpptest/test_vector_init: cpptest/CMakeFiles/test_vector_init.dir/testVectorInit.cpp.o
cpptest/test_vector_init: cpptest/CMakeFiles/test_vector_init.dir/build.make
cpptest/test_vector_init: cpptest/CMakeFiles/test_vector_init.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_vector_init"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_vector_init.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpptest/CMakeFiles/test_vector_init.dir/build: cpptest/test_vector_init

.PHONY : cpptest/CMakeFiles/test_vector_init.dir/build

cpptest/CMakeFiles/test_vector_init.dir/clean:
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && $(CMAKE_COMMAND) -P CMakeFiles/test_vector_init.dir/cmake_clean.cmake
.PHONY : cpptest/CMakeFiles/test_vector_init.dir/clean

cpptest/CMakeFiles/test_vector_init.dir/depend:
	cd /home/songzhen/workspace/ADGNN/cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN/cpptest /home/songzhen/workspace/ADGNN/cmake/build /home/songzhen/workspace/ADGNN/cmake/build/cpptest /home/songzhen/workspace/ADGNN/cmake/build/cpptest/CMakeFiles/test_vector_init.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpptest/CMakeFiles/test_vector_init.dir/depend
