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
include cpptest/CMakeFiles/fundamental_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include cpptest/CMakeFiles/fundamental_test.dir/compiler_depend.make

# Include the progress variables for this target.
include cpptest/CMakeFiles/fundamental_test.dir/progress.make

# Include the compile flags for this target's objects.
include cpptest/CMakeFiles/fundamental_test.dir/flags.make

cpptest/CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.o: cpptest/CMakeFiles/fundamental_test.dir/flags.make
cpptest/CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.o: /home/songzhen/workspace/ADGNN/cpptest/test_cpp/fundamental_test.cpp
cpptest/CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.o: cpptest/CMakeFiles/fundamental_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cpptest/CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.o -MF CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.o.d -o CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.o -c /home/songzhen/workspace/ADGNN/cpptest/test_cpp/fundamental_test.cpp

cpptest/CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/test_cpp/fundamental_test.cpp > CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.i

cpptest/CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/test_cpp/fundamental_test.cpp -o CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.s

# Object files for target fundamental_test
fundamental_test_OBJECTS = \
"CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.o"

# External object files for target fundamental_test
fundamental_test_EXTERNAL_OBJECTS =

cpptest/fundamental_test: cpptest/CMakeFiles/fundamental_test.dir/test_cpp/fundamental_test.cpp.o
cpptest/fundamental_test: cpptest/CMakeFiles/fundamental_test.dir/build.make
cpptest/fundamental_test: cpptest/CMakeFiles/fundamental_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fundamental_test"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fundamental_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpptest/CMakeFiles/fundamental_test.dir/build: cpptest/fundamental_test
.PHONY : cpptest/CMakeFiles/fundamental_test.dir/build

cpptest/CMakeFiles/fundamental_test.dir/clean:
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && $(CMAKE_COMMAND) -P CMakeFiles/fundamental_test.dir/cmake_clean.cmake
.PHONY : cpptest/CMakeFiles/fundamental_test.dir/clean

cpptest/CMakeFiles/fundamental_test.dir/depend:
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN/cpptest /home/songzhen/workspace/ADGNN/cmake-build-debug /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest/CMakeFiles/fundamental_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpptest/CMakeFiles/fundamental_test.dir/depend

