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
include cpptest/CMakeFiles/hashset_test.dir/depend.make

# Include the progress variables for this target.
include cpptest/CMakeFiles/hashset_test.dir/progress.make

# Include the compile flags for this target's objects.
include cpptest/CMakeFiles/hashset_test.dir/flags.make

cpptest/CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.o: cpptest/CMakeFiles/hashset_test.dir/flags.make
cpptest/CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.o: ../../cpptest/test_cpp/hashset_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cpptest/CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.o -c /home/songzhen/workspace/ADGNN/cpptest/test_cpp/hashset_test.cpp

cpptest/CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/test_cpp/hashset_test.cpp > CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.i

cpptest/CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/test_cpp/hashset_test.cpp -o CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.s

# Object files for target hashset_test
hashset_test_OBJECTS = \
"CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.o"

# External object files for target hashset_test
hashset_test_EXTERNAL_OBJECTS =

cpptest/hashset_test: cpptest/CMakeFiles/hashset_test.dir/test_cpp/hashset_test.cpp.o
cpptest/hashset_test: cpptest/CMakeFiles/hashset_test.dir/build.make
cpptest/hashset_test: core/structure/libstructure.a
cpptest/hashset_test: /home/songzhen/anaconda3/envs/python3.6/lib/libpython3.6m.so
cpptest/hashset_test: cpptest/CMakeFiles/hashset_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hashset_test"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hashset_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpptest/CMakeFiles/hashset_test.dir/build: cpptest/hashset_test

.PHONY : cpptest/CMakeFiles/hashset_test.dir/build

cpptest/CMakeFiles/hashset_test.dir/clean:
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && $(CMAKE_COMMAND) -P CMakeFiles/hashset_test.dir/cmake_clean.cmake
.PHONY : cpptest/CMakeFiles/hashset_test.dir/clean

cpptest/CMakeFiles/hashset_test.dir/depend:
	cd /home/songzhen/workspace/ADGNN/cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN/cpptest /home/songzhen/workspace/ADGNN/cmake/build /home/songzhen/workspace/ADGNN/cmake/build/cpptest /home/songzhen/workspace/ADGNN/cmake/build/cpptest/CMakeFiles/hashset_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpptest/CMakeFiles/hashset_test.dir/depend

