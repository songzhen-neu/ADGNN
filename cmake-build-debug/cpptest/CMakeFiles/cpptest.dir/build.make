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
include cpptest/CMakeFiles/cpptest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include cpptest/CMakeFiles/cpptest.dir/compiler_depend.make

# Include the progress variables for this target.
include cpptest/CMakeFiles/cpptest.dir/progress.make

# Include the compile flags for this target's objects.
include cpptest/CMakeFiles/cpptest.dir/flags.make

cpptest/CMakeFiles/cpptest.dir/Animal.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/Animal.cc.o: /home/songzhen/workspace/ADGNN/cpptest/Animal.cc
cpptest/CMakeFiles/cpptest.dir/Animal.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cpptest/CMakeFiles/cpptest.dir/Animal.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/Animal.cc.o -MF CMakeFiles/cpptest.dir/Animal.cc.o.d -o CMakeFiles/cpptest.dir/Animal.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/Animal.cc

cpptest/CMakeFiles/cpptest.dir/Animal.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/Animal.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/Animal.cc > CMakeFiles/cpptest.dir/Animal.cc.i

cpptest/CMakeFiles/cpptest.dir/Animal.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/Animal.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/Animal.cc -o CMakeFiles/cpptest.dir/Animal.cc.s

cpptest/CMakeFiles/cpptest.dir/MultiThread.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/MultiThread.cc.o: /home/songzhen/workspace/ADGNN/cpptest/MultiThread.cc
cpptest/CMakeFiles/cpptest.dir/MultiThread.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object cpptest/CMakeFiles/cpptest.dir/MultiThread.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/MultiThread.cc.o -MF CMakeFiles/cpptest.dir/MultiThread.cc.o.d -o CMakeFiles/cpptest.dir/MultiThread.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/MultiThread.cc

cpptest/CMakeFiles/cpptest.dir/MultiThread.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/MultiThread.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/MultiThread.cc > CMakeFiles/cpptest.dir/MultiThread.cc.i

cpptest/CMakeFiles/cpptest.dir/MultiThread.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/MultiThread.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/MultiThread.cc -o CMakeFiles/cpptest.dir/MultiThread.cc.s

cpptest/CMakeFiles/cpptest.dir/address_or_value.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/address_or_value.cc.o: /home/songzhen/workspace/ADGNN/cpptest/address_or_value.cc
cpptest/CMakeFiles/cpptest.dir/address_or_value.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object cpptest/CMakeFiles/cpptest.dir/address_or_value.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/address_or_value.cc.o -MF CMakeFiles/cpptest.dir/address_or_value.cc.o.d -o CMakeFiles/cpptest.dir/address_or_value.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/address_or_value.cc

cpptest/CMakeFiles/cpptest.dir/address_or_value.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/address_or_value.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/address_or_value.cc > CMakeFiles/cpptest.dir/address_or_value.cc.i

cpptest/CMakeFiles/cpptest.dir/address_or_value.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/address_or_value.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/address_or_value.cc -o CMakeFiles/cpptest.dir/address_or_value.cc.s

cpptest/CMakeFiles/cpptest.dir/array_variable_test.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/array_variable_test.cc.o: /home/songzhen/workspace/ADGNN/cpptest/array_variable_test.cc
cpptest/CMakeFiles/cpptest.dir/array_variable_test.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object cpptest/CMakeFiles/cpptest.dir/array_variable_test.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/array_variable_test.cc.o -MF CMakeFiles/cpptest.dir/array_variable_test.cc.o.d -o CMakeFiles/cpptest.dir/array_variable_test.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/array_variable_test.cc

cpptest/CMakeFiles/cpptest.dir/array_variable_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/array_variable_test.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/array_variable_test.cc > CMakeFiles/cpptest.dir/array_variable_test.cc.i

cpptest/CMakeFiles/cpptest.dir/array_variable_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/array_variable_test.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/array_variable_test.cc -o CMakeFiles/cpptest.dir/array_variable_test.cc.s

cpptest/CMakeFiles/cpptest.dir/atomic_test.cpp.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/atomic_test.cpp.o: /home/songzhen/workspace/ADGNN/cpptest/atomic_test.cpp
cpptest/CMakeFiles/cpptest.dir/atomic_test.cpp.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object cpptest/CMakeFiles/cpptest.dir/atomic_test.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/atomic_test.cpp.o -MF CMakeFiles/cpptest.dir/atomic_test.cpp.o.d -o CMakeFiles/cpptest.dir/atomic_test.cpp.o -c /home/songzhen/workspace/ADGNN/cpptest/atomic_test.cpp

cpptest/CMakeFiles/cpptest.dir/atomic_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/atomic_test.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/atomic_test.cpp > CMakeFiles/cpptest.dir/atomic_test.cpp.i

cpptest/CMakeFiles/cpptest.dir/atomic_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/atomic_test.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/atomic_test.cpp -o CMakeFiles/cpptest.dir/atomic_test.cpp.s

cpptest/CMakeFiles/cpptest.dir/bittest.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/bittest.cc.o: /home/songzhen/workspace/ADGNN/cpptest/bittest.cc
cpptest/CMakeFiles/cpptest.dir/bittest.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object cpptest/CMakeFiles/cpptest.dir/bittest.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/bittest.cc.o -MF CMakeFiles/cpptest.dir/bittest.cc.o.d -o CMakeFiles/cpptest.dir/bittest.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/bittest.cc

cpptest/CMakeFiles/cpptest.dir/bittest.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/bittest.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/bittest.cc > CMakeFiles/cpptest.dir/bittest.cc.i

cpptest/CMakeFiles/cpptest.dir/bittest.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/bittest.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/bittest.cc -o CMakeFiles/cpptest.dir/bittest.cc.s

cpptest/CMakeFiles/cpptest.dir/copy_test.cpp.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/copy_test.cpp.o: /home/songzhen/workspace/ADGNN/cpptest/copy_test.cpp
cpptest/CMakeFiles/cpptest.dir/copy_test.cpp.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object cpptest/CMakeFiles/cpptest.dir/copy_test.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/copy_test.cpp.o -MF CMakeFiles/cpptest.dir/copy_test.cpp.o.d -o CMakeFiles/cpptest.dir/copy_test.cpp.o -c /home/songzhen/workspace/ADGNN/cpptest/copy_test.cpp

cpptest/CMakeFiles/cpptest.dir/copy_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/copy_test.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/copy_test.cpp > CMakeFiles/cpptest.dir/copy_test.cpp.i

cpptest/CMakeFiles/cpptest.dir/copy_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/copy_test.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/copy_test.cpp -o CMakeFiles/cpptest.dir/copy_test.cpp.s

cpptest/CMakeFiles/cpptest.dir/get_length.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/get_length.cc.o: /home/songzhen/workspace/ADGNN/cpptest/get_length.cc
cpptest/CMakeFiles/cpptest.dir/get_length.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object cpptest/CMakeFiles/cpptest.dir/get_length.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/get_length.cc.o -MF CMakeFiles/cpptest.dir/get_length.cc.o.d -o CMakeFiles/cpptest.dir/get_length.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/get_length.cc

cpptest/CMakeFiles/cpptest.dir/get_length.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/get_length.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/get_length.cc > CMakeFiles/cpptest.dir/get_length.cc.i

cpptest/CMakeFiles/cpptest.dir/get_length.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/get_length.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/get_length.cc -o CMakeFiles/cpptest.dir/get_length.cc.s

cpptest/CMakeFiles/cpptest.dir/map_insert_test.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/map_insert_test.cc.o: /home/songzhen/workspace/ADGNN/cpptest/map_insert_test.cc
cpptest/CMakeFiles/cpptest.dir/map_insert_test.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object cpptest/CMakeFiles/cpptest.dir/map_insert_test.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/map_insert_test.cc.o -MF CMakeFiles/cpptest.dir/map_insert_test.cc.o.d -o CMakeFiles/cpptest.dir/map_insert_test.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/map_insert_test.cc

cpptest/CMakeFiles/cpptest.dir/map_insert_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/map_insert_test.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/map_insert_test.cc > CMakeFiles/cpptest.dir/map_insert_test.cc.i

cpptest/CMakeFiles/cpptest.dir/map_insert_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/map_insert_test.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/map_insert_test.cc -o CMakeFiles/cpptest.dir/map_insert_test.cc.s

cpptest/CMakeFiles/cpptest.dir/pointer_test.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/pointer_test.cc.o: /home/songzhen/workspace/ADGNN/cpptest/pointer_test.cc
cpptest/CMakeFiles/cpptest.dir/pointer_test.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object cpptest/CMakeFiles/cpptest.dir/pointer_test.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/pointer_test.cc.o -MF CMakeFiles/cpptest.dir/pointer_test.cc.o.d -o CMakeFiles/cpptest.dir/pointer_test.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/pointer_test.cc

cpptest/CMakeFiles/cpptest.dir/pointer_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/pointer_test.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/pointer_test.cc > CMakeFiles/cpptest.dir/pointer_test.cc.i

cpptest/CMakeFiles/cpptest.dir/pointer_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/pointer_test.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/pointer_test.cc -o CMakeFiles/cpptest.dir/pointer_test.cc.s

cpptest/CMakeFiles/cpptest.dir/startMaster.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/startMaster.cc.o: /home/songzhen/workspace/ADGNN/cpptest/startMaster.cc
cpptest/CMakeFiles/cpptest.dir/startMaster.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object cpptest/CMakeFiles/cpptest.dir/startMaster.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/startMaster.cc.o -MF CMakeFiles/cpptest.dir/startMaster.cc.o.d -o CMakeFiles/cpptest.dir/startMaster.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/startMaster.cc

cpptest/CMakeFiles/cpptest.dir/startMaster.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/startMaster.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/startMaster.cc > CMakeFiles/cpptest.dir/startMaster.cc.i

cpptest/CMakeFiles/cpptest.dir/startMaster.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/startMaster.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/startMaster.cc -o CMakeFiles/cpptest.dir/startMaster.cc.s

cpptest/CMakeFiles/cpptest.dir/staticTest.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/staticTest.cc.o: /home/songzhen/workspace/ADGNN/cpptest/staticTest.cc
cpptest/CMakeFiles/cpptest.dir/staticTest.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object cpptest/CMakeFiles/cpptest.dir/staticTest.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/staticTest.cc.o -MF CMakeFiles/cpptest.dir/staticTest.cc.o.d -o CMakeFiles/cpptest.dir/staticTest.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/staticTest.cc

cpptest/CMakeFiles/cpptest.dir/staticTest.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/staticTest.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/staticTest.cc > CMakeFiles/cpptest.dir/staticTest.cc.i

cpptest/CMakeFiles/cpptest.dir/staticTest.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/staticTest.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/staticTest.cc -o CMakeFiles/cpptest.dir/staticTest.cc.s

cpptest/CMakeFiles/cpptest.dir/testIterator.cpp.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/testIterator.cpp.o: /home/songzhen/workspace/ADGNN/cpptest/testIterator.cpp
cpptest/CMakeFiles/cpptest.dir/testIterator.cpp.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object cpptest/CMakeFiles/cpptest.dir/testIterator.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/testIterator.cpp.o -MF CMakeFiles/cpptest.dir/testIterator.cpp.o.d -o CMakeFiles/cpptest.dir/testIterator.cpp.o -c /home/songzhen/workspace/ADGNN/cpptest/testIterator.cpp

cpptest/CMakeFiles/cpptest.dir/testIterator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/testIterator.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/testIterator.cpp > CMakeFiles/cpptest.dir/testIterator.cpp.i

cpptest/CMakeFiles/cpptest.dir/testIterator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/testIterator.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/testIterator.cpp -o CMakeFiles/cpptest.dir/testIterator.cpp.s

cpptest/CMakeFiles/cpptest.dir/testVectorInit.cpp.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/testVectorInit.cpp.o: /home/songzhen/workspace/ADGNN/cpptest/testVectorInit.cpp
cpptest/CMakeFiles/cpptest.dir/testVectorInit.cpp.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object cpptest/CMakeFiles/cpptest.dir/testVectorInit.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/testVectorInit.cpp.o -MF CMakeFiles/cpptest.dir/testVectorInit.cpp.o.d -o CMakeFiles/cpptest.dir/testVectorInit.cpp.o -c /home/songzhen/workspace/ADGNN/cpptest/testVectorInit.cpp

cpptest/CMakeFiles/cpptest.dir/testVectorInit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/testVectorInit.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/testVectorInit.cpp > CMakeFiles/cpptest.dir/testVectorInit.cpp.i

cpptest/CMakeFiles/cpptest.dir/testVectorInit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/testVectorInit.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/testVectorInit.cpp -o CMakeFiles/cpptest.dir/testVectorInit.cpp.s

cpptest/CMakeFiles/cpptest.dir/test_condition_variable.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/test_condition_variable.cc.o: /home/songzhen/workspace/ADGNN/cpptest/test_condition_variable.cc
cpptest/CMakeFiles/cpptest.dir/test_condition_variable.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object cpptest/CMakeFiles/cpptest.dir/test_condition_variable.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/test_condition_variable.cc.o -MF CMakeFiles/cpptest.dir/test_condition_variable.cc.o.d -o CMakeFiles/cpptest.dir/test_condition_variable.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/test_condition_variable.cc

cpptest/CMakeFiles/cpptest.dir/test_condition_variable.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/test_condition_variable.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/test_condition_variable.cc > CMakeFiles/cpptest.dir/test_condition_variable.cc.i

cpptest/CMakeFiles/cpptest.dir/test_condition_variable.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/test_condition_variable.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/test_condition_variable.cc -o CMakeFiles/cpptest.dir/test_condition_variable.cc.s

cpptest/CMakeFiles/cpptest.dir/test_matrix_multi.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/test_matrix_multi.cc.o: /home/songzhen/workspace/ADGNN/cpptest/test_matrix_multi.cc
cpptest/CMakeFiles/cpptest.dir/test_matrix_multi.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object cpptest/CMakeFiles/cpptest.dir/test_matrix_multi.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/test_matrix_multi.cc.o -MF CMakeFiles/cpptest.dir/test_matrix_multi.cc.o.d -o CMakeFiles/cpptest.dir/test_matrix_multi.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/test_matrix_multi.cc

cpptest/CMakeFiles/cpptest.dir/test_matrix_multi.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/test_matrix_multi.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/test_matrix_multi.cc > CMakeFiles/cpptest.dir/test_matrix_multi.cc.i

cpptest/CMakeFiles/cpptest.dir/test_matrix_multi.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/test_matrix_multi.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/test_matrix_multi.cc -o CMakeFiles/cpptest.dir/test_matrix_multi.cc.s

cpptest/CMakeFiles/cpptest.dir/test_mutex.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/test_mutex.cc.o: /home/songzhen/workspace/ADGNN/cpptest/test_mutex.cc
cpptest/CMakeFiles/cpptest.dir/test_mutex.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object cpptest/CMakeFiles/cpptest.dir/test_mutex.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/test_mutex.cc.o -MF CMakeFiles/cpptest.dir/test_mutex.cc.o.d -o CMakeFiles/cpptest.dir/test_mutex.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/test_mutex.cc

cpptest/CMakeFiles/cpptest.dir/test_mutex.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/test_mutex.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/test_mutex.cc > CMakeFiles/cpptest.dir/test_mutex.cc.i

cpptest/CMakeFiles/cpptest.dir/test_mutex.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/test_mutex.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/test_mutex.cc -o CMakeFiles/cpptest.dir/test_mutex.cc.s

cpptest/CMakeFiles/cpptest.dir/test_wait.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/test_wait.cc.o: /home/songzhen/workspace/ADGNN/cpptest/test_wait.cc
cpptest/CMakeFiles/cpptest.dir/test_wait.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building CXX object cpptest/CMakeFiles/cpptest.dir/test_wait.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/test_wait.cc.o -MF CMakeFiles/cpptest.dir/test_wait.cc.o.d -o CMakeFiles/cpptest.dir/test_wait.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/test_wait.cc

cpptest/CMakeFiles/cpptest.dir/test_wait.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/test_wait.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/test_wait.cc > CMakeFiles/cpptest.dir/test_wait.cc.i

cpptest/CMakeFiles/cpptest.dir/test_wait.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/test_wait.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/test_wait.cc -o CMakeFiles/cpptest.dir/test_wait.cc.s

cpptest/CMakeFiles/cpptest.dir/vector_test.cc.o: cpptest/CMakeFiles/cpptest.dir/flags.make
cpptest/CMakeFiles/cpptest.dir/vector_test.cc.o: /home/songzhen/workspace/ADGNN/cpptest/vector_test.cc
cpptest/CMakeFiles/cpptest.dir/vector_test.cc.o: cpptest/CMakeFiles/cpptest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Building CXX object cpptest/CMakeFiles/cpptest.dir/vector_test.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpptest/CMakeFiles/cpptest.dir/vector_test.cc.o -MF CMakeFiles/cpptest.dir/vector_test.cc.o.d -o CMakeFiles/cpptest.dir/vector_test.cc.o -c /home/songzhen/workspace/ADGNN/cpptest/vector_test.cc

cpptest/CMakeFiles/cpptest.dir/vector_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptest.dir/vector_test.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/cpptest/vector_test.cc > CMakeFiles/cpptest.dir/vector_test.cc.i

cpptest/CMakeFiles/cpptest.dir/vector_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptest.dir/vector_test.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/cpptest/vector_test.cc -o CMakeFiles/cpptest.dir/vector_test.cc.s

# Object files for target cpptest
cpptest_OBJECTS = \
"CMakeFiles/cpptest.dir/Animal.cc.o" \
"CMakeFiles/cpptest.dir/MultiThread.cc.o" \
"CMakeFiles/cpptest.dir/address_or_value.cc.o" \
"CMakeFiles/cpptest.dir/array_variable_test.cc.o" \
"CMakeFiles/cpptest.dir/atomic_test.cpp.o" \
"CMakeFiles/cpptest.dir/bittest.cc.o" \
"CMakeFiles/cpptest.dir/copy_test.cpp.o" \
"CMakeFiles/cpptest.dir/get_length.cc.o" \
"CMakeFiles/cpptest.dir/map_insert_test.cc.o" \
"CMakeFiles/cpptest.dir/pointer_test.cc.o" \
"CMakeFiles/cpptest.dir/startMaster.cc.o" \
"CMakeFiles/cpptest.dir/staticTest.cc.o" \
"CMakeFiles/cpptest.dir/testIterator.cpp.o" \
"CMakeFiles/cpptest.dir/testVectorInit.cpp.o" \
"CMakeFiles/cpptest.dir/test_condition_variable.cc.o" \
"CMakeFiles/cpptest.dir/test_matrix_multi.cc.o" \
"CMakeFiles/cpptest.dir/test_mutex.cc.o" \
"CMakeFiles/cpptest.dir/test_wait.cc.o" \
"CMakeFiles/cpptest.dir/vector_test.cc.o"

# External object files for target cpptest
cpptest_EXTERNAL_OBJECTS =

lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/Animal.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/MultiThread.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/address_or_value.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/array_variable_test.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/atomic_test.cpp.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/bittest.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/copy_test.cpp.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/get_length.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/map_insert_test.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/pointer_test.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/startMaster.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/staticTest.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/testIterator.cpp.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/testVectorInit.cpp.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/test_condition_variable.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/test_matrix_multi.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/test_mutex.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/test_wait.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/vector_test.cc.o
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/build.make
lib/libcpptest.a: cpptest/CMakeFiles/cpptest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ADGNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_20) "Linking CXX static library ../lib/libcpptest.a"
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && $(CMAKE_COMMAND) -P CMakeFiles/cpptest.dir/cmake_clean_target.cmake
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpptest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpptest/CMakeFiles/cpptest.dir/build: lib/libcpptest.a
.PHONY : cpptest/CMakeFiles/cpptest.dir/build

cpptest/CMakeFiles/cpptest.dir/clean:
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest && $(CMAKE_COMMAND) -P CMakeFiles/cpptest.dir/cmake_clean.cmake
.PHONY : cpptest/CMakeFiles/cpptest.dir/clean

cpptest/CMakeFiles/cpptest.dir/depend:
	cd /home/songzhen/workspace/ADGNN/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN/cpptest /home/songzhen/workspace/ADGNN/cmake-build-debug /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest /home/songzhen/workspace/ADGNN/cmake-build-debug/cpptest/CMakeFiles/cpptest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpptest/CMakeFiles/cpptest.dir/depend

