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
include core/structure/CMakeFiles/structure.dir/depend.make

# Include the progress variables for this target.
include core/structure/CMakeFiles/structure.dir/progress.make

# Include the compile flags for this target's objects.
include core/structure/CMakeFiles/structure.dir/flags.make

core/structure/CMakeFiles/structure.dir/Graph.cpp.o: core/structure/CMakeFiles/structure.dir/flags.make
core/structure/CMakeFiles/structure.dir/Graph.cpp.o: ../../core/structure/Graph.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object core/structure/CMakeFiles/structure.dir/Graph.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/structure.dir/Graph.cpp.o -c /home/songzhen/workspace/ADGNN/core/structure/Graph.cpp

core/structure/CMakeFiles/structure.dir/Graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/structure.dir/Graph.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/core/structure/Graph.cpp > CMakeFiles/structure.dir/Graph.cpp.i

core/structure/CMakeFiles/structure.dir/Graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/structure.dir/Graph.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/core/structure/Graph.cpp -o CMakeFiles/structure.dir/Graph.cpp.s

core/structure/CMakeFiles/structure.dir/GraphLayer.cpp.o: core/structure/CMakeFiles/structure.dir/flags.make
core/structure/CMakeFiles/structure.dir/GraphLayer.cpp.o: ../../core/structure/GraphLayer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object core/structure/CMakeFiles/structure.dir/GraphLayer.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/structure.dir/GraphLayer.cpp.o -c /home/songzhen/workspace/ADGNN/core/structure/GraphLayer.cpp

core/structure/CMakeFiles/structure.dir/GraphLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/structure.dir/GraphLayer.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/core/structure/GraphLayer.cpp > CMakeFiles/structure.dir/GraphLayer.cpp.i

core/structure/CMakeFiles/structure.dir/GraphLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/structure.dir/GraphLayer.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/core/structure/GraphLayer.cpp -o CMakeFiles/structure.dir/GraphLayer.cpp.s

core/structure/CMakeFiles/structure.dir/SubGraph.cpp.o: core/structure/CMakeFiles/structure.dir/flags.make
core/structure/CMakeFiles/structure.dir/SubGraph.cpp.o: ../../core/structure/SubGraph.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object core/structure/CMakeFiles/structure.dir/SubGraph.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/structure.dir/SubGraph.cpp.o -c /home/songzhen/workspace/ADGNN/core/structure/SubGraph.cpp

core/structure/CMakeFiles/structure.dir/SubGraph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/structure.dir/SubGraph.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/core/structure/SubGraph.cpp > CMakeFiles/structure.dir/SubGraph.cpp.i

core/structure/CMakeFiles/structure.dir/SubGraph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/structure.dir/SubGraph.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/core/structure/SubGraph.cpp -o CMakeFiles/structure.dir/SubGraph.cpp.s

core/structure/CMakeFiles/structure.dir/hashmap.cpp.o: core/structure/CMakeFiles/structure.dir/flags.make
core/structure/CMakeFiles/structure.dir/hashmap.cpp.o: ../../core/structure/hashmap.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object core/structure/CMakeFiles/structure.dir/hashmap.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/structure.dir/hashmap.cpp.o -c /home/songzhen/workspace/ADGNN/core/structure/hashmap.cpp

core/structure/CMakeFiles/structure.dir/hashmap.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/structure.dir/hashmap.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/core/structure/hashmap.cpp > CMakeFiles/structure.dir/hashmap.cpp.i

core/structure/CMakeFiles/structure.dir/hashmap.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/structure.dir/hashmap.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/core/structure/hashmap.cpp -o CMakeFiles/structure.dir/hashmap.cpp.s

core/structure/CMakeFiles/structure.dir/hashset.cpp.o: core/structure/CMakeFiles/structure.dir/flags.make
core/structure/CMakeFiles/structure.dir/hashset.cpp.o: ../../core/structure/hashset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object core/structure/CMakeFiles/structure.dir/hashset.cpp.o"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/structure.dir/hashset.cpp.o -c /home/songzhen/workspace/ADGNN/core/structure/hashset.cpp

core/structure/CMakeFiles/structure.dir/hashset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/structure.dir/hashset.cpp.i"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/core/structure/hashset.cpp > CMakeFiles/structure.dir/hashset.cpp.i

core/structure/CMakeFiles/structure.dir/hashset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/structure.dir/hashset.cpp.s"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/core/structure/hashset.cpp -o CMakeFiles/structure.dir/hashset.cpp.s

# Object files for target structure
structure_OBJECTS = \
"CMakeFiles/structure.dir/Graph.cpp.o" \
"CMakeFiles/structure.dir/GraphLayer.cpp.o" \
"CMakeFiles/structure.dir/SubGraph.cpp.o" \
"CMakeFiles/structure.dir/hashmap.cpp.o" \
"CMakeFiles/structure.dir/hashset.cpp.o"

# External object files for target structure
structure_EXTERNAL_OBJECTS =

core/structure/libstructure.a: core/structure/CMakeFiles/structure.dir/Graph.cpp.o
core/structure/libstructure.a: core/structure/CMakeFiles/structure.dir/GraphLayer.cpp.o
core/structure/libstructure.a: core/structure/CMakeFiles/structure.dir/SubGraph.cpp.o
core/structure/libstructure.a: core/structure/CMakeFiles/structure.dir/hashmap.cpp.o
core/structure/libstructure.a: core/structure/CMakeFiles/structure.dir/hashset.cpp.o
core/structure/libstructure.a: core/structure/CMakeFiles/structure.dir/build.make
core/structure/libstructure.a: core/structure/CMakeFiles/structure.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library libstructure.a"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && $(CMAKE_COMMAND) -P CMakeFiles/structure.dir/cmake_clean_target.cmake
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/structure.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
core/structure/CMakeFiles/structure.dir/build: core/structure/libstructure.a

.PHONY : core/structure/CMakeFiles/structure.dir/build

core/structure/CMakeFiles/structure.dir/clean:
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/structure && $(CMAKE_COMMAND) -P CMakeFiles/structure.dir/cmake_clean.cmake
.PHONY : core/structure/CMakeFiles/structure.dir/clean

core/structure/CMakeFiles/structure.dir/depend:
	cd /home/songzhen/workspace/ADGNN/cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN/core/structure /home/songzhen/workspace/ADGNN/cmake/build /home/songzhen/workspace/ADGNN/cmake/build/core/structure /home/songzhen/workspace/ADGNN/cmake/build/core/structure/CMakeFiles/structure.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : core/structure/CMakeFiles/structure.dir/depend

