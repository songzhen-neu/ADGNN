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
include CMakeFiles/mainTest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mainTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mainTest.dir/flags.make

CMakeFiles/mainTest.dir/main.cc.o: CMakeFiles/mainTest.dir/flags.make
CMakeFiles/mainTest.dir/main.cc.o: ../../main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mainTest.dir/main.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mainTest.dir/main.cc.o -c /home/songzhen/workspace/ADGNN/main.cc

CMakeFiles/mainTest.dir/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mainTest.dir/main.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/main.cc > CMakeFiles/mainTest.dir/main.cc.i

CMakeFiles/mainTest.dir/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mainTest.dir/main.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/main.cc -o CMakeFiles/mainTest.dir/main.cc.s

# Object files for target mainTest
mainTest_OBJECTS = \
"CMakeFiles/mainTest.dir/main.cc.o"

# External object files for target mainTest
mainTest_EXTERNAL_OBJECTS =

mainTest: CMakeFiles/mainTest.dir/main.cc.o
mainTest: CMakeFiles/mainTest.dir/build.make
mainTest: core/service/libservice.a
mainTest: core/util/libutil.a
mainTest: core/store/libstore.a
mainTest: core/partition/libpartition.a
mainTest: core/service/libservice.a
mainTest: core/util/libutil.a
mainTest: core/store/libstore.a
mainTest: core/partition/libpartition.a
mainTest: libhw_grpc_proto.a
mainTest: /usr/local/lib/libgrpc++_reflection.a
mainTest: /usr/local/lib/libgrpc++.a
mainTest: /usr/local/lib/libgrpc.a
mainTest: /usr/local/lib/libz.a
mainTest: /usr/local/lib/libcares.a
mainTest: /usr/local/lib/libaddress_sorting.a
mainTest: /usr/local/lib/libre2.a
mainTest: /usr/local/lib/libabsl_raw_hash_set.a
mainTest: /usr/local/lib/libabsl_hashtablez_sampler.a
mainTest: /usr/local/lib/libabsl_hash.a
mainTest: /usr/local/lib/libabsl_city.a
mainTest: /usr/local/lib/libabsl_low_level_hash.a
mainTest: /usr/local/lib/libabsl_statusor.a
mainTest: /usr/local/lib/libabsl_bad_variant_access.a
mainTest: /usr/local/lib/libgpr.a
mainTest: /usr/local/lib/libupb.a
mainTest: /usr/local/lib/libabsl_status.a
mainTest: /usr/local/lib/libabsl_random_distributions.a
mainTest: /usr/local/lib/libabsl_random_seed_sequences.a
mainTest: /usr/local/lib/libabsl_random_internal_pool_urbg.a
mainTest: /usr/local/lib/libabsl_random_internal_randen.a
mainTest: /usr/local/lib/libabsl_random_internal_randen_hwaes.a
mainTest: /usr/local/lib/libabsl_random_internal_randen_hwaes_impl.a
mainTest: /usr/local/lib/libabsl_random_internal_randen_slow.a
mainTest: /usr/local/lib/libabsl_random_internal_platform.a
mainTest: /usr/local/lib/libabsl_random_internal_seed_material.a
mainTest: /usr/local/lib/libabsl_random_seed_gen_exception.a
mainTest: /usr/local/lib/libabsl_cord.a
mainTest: /usr/local/lib/libabsl_bad_optional_access.a
mainTest: /usr/local/lib/libabsl_cordz_info.a
mainTest: /usr/local/lib/libabsl_cord_internal.a
mainTest: /usr/local/lib/libabsl_cordz_functions.a
mainTest: /usr/local/lib/libabsl_exponential_biased.a
mainTest: /usr/local/lib/libabsl_cordz_handle.a
mainTest: /usr/local/lib/libabsl_str_format_internal.a
mainTest: /usr/local/lib/libabsl_synchronization.a
mainTest: /usr/local/lib/libabsl_stacktrace.a
mainTest: /usr/local/lib/libabsl_symbolize.a
mainTest: /usr/local/lib/libabsl_debugging_internal.a
mainTest: /usr/local/lib/libabsl_demangle_internal.a
mainTest: /usr/local/lib/libabsl_graphcycles_internal.a
mainTest: /usr/local/lib/libabsl_malloc_internal.a
mainTest: /usr/local/lib/libabsl_time.a
mainTest: /usr/local/lib/libabsl_strings.a
mainTest: /usr/local/lib/libabsl_throw_delegate.a
mainTest: /usr/local/lib/libabsl_int128.a
mainTest: /usr/local/lib/libabsl_strings_internal.a
mainTest: /usr/local/lib/libabsl_base.a
mainTest: /usr/local/lib/libabsl_spinlock_wait.a
mainTest: /usr/local/lib/libabsl_raw_logging_internal.a
mainTest: /usr/local/lib/libabsl_log_severity.a
mainTest: /usr/local/lib/libabsl_civil_time.a
mainTest: /usr/local/lib/libabsl_time_zone.a
mainTest: /usr/local/lib/libssl.a
mainTest: /usr/local/lib/libcrypto.a
mainTest: /usr/local/lib/libprotobuf.a
mainTest: core/structure/libstructure.a
mainTest: /home/songzhen/anaconda3/envs/python3.6/lib/libpython3.6m.so
mainTest: CMakeFiles/mainTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mainTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mainTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mainTest.dir/build: mainTest

.PHONY : CMakeFiles/mainTest.dir/build

CMakeFiles/mainTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mainTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mainTest.dir/clean

CMakeFiles/mainTest.dir/depend:
	cd /home/songzhen/workspace/ADGNN/cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN/cmake/build /home/songzhen/workspace/ADGNN/cmake/build /home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles/mainTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mainTest.dir/depend

