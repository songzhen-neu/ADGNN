# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

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
CMAKE_COMMAND = /home/songzhen/Documents/CLion-2020.2.3/clion-2020.2.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/songzhen/Documents/CLion-2020.2.3/clion-2020.2.3/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/songzhen/workspace/ec-graph

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/songzhen/workspace/ec-graph/cmake-build-debug

# Include any dependencies generated for this target.
include cpptest/CMakeFiles/copy.dir/depend.make

# Include the progress variables for this target.
include cpptest/CMakeFiles/copy.dir/progress.make

# Include the compile flags for this target's objects.
include cpptest/CMakeFiles/copy.dir/flags.make

cpptest/CMakeFiles/copy.dir/address_or_value.cc.o: cpptest/CMakeFiles/copy.dir/flags.make
cpptest/CMakeFiles/copy.dir/address_or_value.cc.o: ../cpptest/address_or_value.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ec-graph/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cpptest/CMakeFiles/copy.dir/address_or_value.cc.o"
	cd /home/songzhen/workspace/ec-graph/cmake-build-debug/cpptest && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/copy.dir/address_or_value.cc.o -c /home/songzhen/workspace/ec-graph/cpptest/address_or_value.cc

cpptest/CMakeFiles/copy.dir/address_or_value.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/copy.dir/address_or_value.cc.i"
	cd /home/songzhen/workspace/ec-graph/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ec-graph/cpptest/address_or_value.cc > CMakeFiles/copy.dir/address_or_value.cc.i

cpptest/CMakeFiles/copy.dir/address_or_value.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/copy.dir/address_or_value.cc.s"
	cd /home/songzhen/workspace/ec-graph/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ec-graph/cpptest/address_or_value.cc -o CMakeFiles/copy.dir/address_or_value.cc.s

# Object files for target copy
copy_OBJECTS = \
"CMakeFiles/copy.dir/address_or_value.cc.o"

# External object files for target copy
copy_EXTERNAL_OBJECTS =

cpptest/copy: cpptest/CMakeFiles/copy.dir/address_or_value.cc.o
cpptest/copy: cpptest/CMakeFiles/copy.dir/build.make
cpptest/copy: cpptest/CMakeFiles/copy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ec-graph/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable copy"
	cd /home/songzhen/workspace/ec-graph/cmake-build-debug/cpptest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/copy.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpptest/CMakeFiles/copy.dir/build: cpptest/copy

.PHONY : cpptest/CMakeFiles/copy.dir/build

cpptest/CMakeFiles/copy.dir/clean:
	cd /home/songzhen/workspace/ec-graph/cmake-build-debug/cpptest && $(CMAKE_COMMAND) -P CMakeFiles/copy.dir/cmake_clean.cmake
.PHONY : cpptest/CMakeFiles/copy.dir/clean

cpptest/CMakeFiles/copy.dir/depend:
	cd /home/songzhen/workspace/ec-graph/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ec-graph /home/songzhen/workspace/ec-graph/cpptest /home/songzhen/workspace/ec-graph/cmake-build-debug /home/songzhen/workspace/ec-graph/cmake-build-debug/cpptest /home/songzhen/workspace/ec-graph/cmake-build-debug/cpptest/CMakeFiles/copy.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpptest/CMakeFiles/copy.dir/depend

