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
include cpptest/CMakeFiles/MultiThread.dir/depend.make

# Include the progress variables for this target.
include cpptest/CMakeFiles/MultiThread.dir/progress.make

# Include the compile flags for this target's objects.
include cpptest/CMakeFiles/MultiThread.dir/flags.make

cpptest/CMakeFiles/MultiThread.dir/MultiThread.cc.o: cpptest/CMakeFiles/MultiThread.dir/flags.make
cpptest/CMakeFiles/MultiThread.dir/MultiThread.cc.o: ../../cpptest/MultiThread.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cpptest/CMakeFiles/MultiThread.dir/MultiThread.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MultiThread.dir/MultiThread.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/MultiThread.cc

cpptest/CMakeFiles/MultiThread.dir/MultiThread.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MultiThread.dir/MultiThread.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/MultiThread.cc > CMakeFiles/MultiThread.dir/MultiThread.cc.i

cpptest/CMakeFiles/MultiThread.dir/MultiThread.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MultiThread.dir/MultiThread.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/MultiThread.cc -o CMakeFiles/MultiThread.dir/MultiThread.cc.s

# Object files for target MultiThread
MultiThread_OBJECTS = \
"CMakeFiles/MultiThread.dir/MultiThread.cc.o"

# External object files for target MultiThread
MultiThread_EXTERNAL_OBJECTS =

cpptest/MultiThread: cpptest/CMakeFiles/MultiThread.dir/MultiThread.cc.o
cpptest/MultiThread: cpptest/CMakeFiles/MultiThread.dir/build.make
cpptest/MultiThread: cpptest/CMakeFiles/MultiThread.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable MultiThread"
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MultiThread.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpptest/CMakeFiles/MultiThread.dir/build: cpptest/MultiThread

.PHONY : cpptest/CMakeFiles/MultiThread.dir/build

cpptest/CMakeFiles/MultiThread.dir/clean:
	cd /home/songzhen/workspace/ADGNN/cmake/build/cpptest && $(CMAKE_COMMAND) -P CMakeFiles/MultiThread.dir/cmake_clean.cmake
.PHONY : cpptest/CMakeFiles/MultiThread.dir/clean

cpptest/CMakeFiles/MultiThread.dir/depend:
	cd /home/songzhen/workspace/ADGNN/cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN/cpptest /home/songzhen/workspace/ADGNN/cmake/build /home/songzhen/workspace/ADGNN/cmake/build/cpptest /home/songzhen/workspace/ADGNN/cmake/build/cpptest/CMakeFiles/MultiThread.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpptest/CMakeFiles/MultiThread.dir/depend
