# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /home/songzhen/Documents/CLion-2023.2.2/clion-2023.2.2/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /home/songzhen/Documents/CLion-2023.2.2/clion-2023.2.2/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/songzhen/workspace/ADGNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/songzhen/workspace/ADGNN/cmake-build-debug

# Include any dependencies generated for this target.
include cpptest/CMakeFiles/get_length.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include cpptest/CMakeFiles/get_length.dir/compiler_depend.make

# Include the progress variables for this target.
include cpptest/CMakeFiles/get_length.dir/progress.make

# Include the compile flags for this target's objects.
include cpptest/CMakeFiles/get_length.dir/flags.make

cpptest/CMakeFiles/get_length.dir/get_length.cc.o: cpptest/CMakeFiles/get_length.dir/flags.make
cpptest/CMakeFiles/get_length.dir/get_length.cc.o: /home/songzhen/workspace/ADGNN/cpptest/get_length.cc
cpptest/CMakeFiles/get_length.dir/get_length.cc.o: cpptest/CMakeFiles/get_length.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cpptest/CMakeFiles/get_length.dir/get_length.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/get_length.dir/get_length.cc.o -MF CMakeFiles/get_length.dir/get_length.cc.o.d -o CMakeFiles/get_length.dir/get_length.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/get_length.cc

cpptest/CMakeFiles/get_length.dir/get_length.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/get_length.dir/get_length.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/get_length.cc > CMakeFiles/get_length.dir/get_length.cc.i

cpptest/CMakeFiles/get_length.dir/get_length.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/get_length.dir/get_length.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/get_length.cc -o CMakeFiles/get_length.dir/get_length.cc.s

# Object files for target get_length
get_length_OBJECTS = \
"CMakeFiles/get_length.dir/get_length.cc.o"

# External object files for target get_length
get_length_EXTERNAL_OBJECTS =

cpptest/get_length: cpptest/CMakeFiles/get_length.dir/get_length.cc.o
cpptest/get_length: cpptest/CMakeFiles/get_length.dir/build.make
cpptest/get_length: cpptest/CMakeFiles/get_length.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable get_length"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/get_length.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpptest/CMakeFiles/get_length.dir/build: cpptest/get_length
.PHONY : cpptest/CMakeFiles/get_length.dir/build

cpptest/CMakeFiles/get_length.dir/clean:
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && $(CMAKE_COMMAND) -P CMakeFiles/get_length.dir/cmake_clean.cmake
.PHONY : cpptest/CMakeFiles/get_length.dir/clean

cpptest/CMakeFiles/get_length.dir/depend:
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN/cpptest /home/songzhen/workspace/ADGNN/cmake-build-debug /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest/CMakeFiles/get_length.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpptest/CMakeFiles/get_length.dir/depend

