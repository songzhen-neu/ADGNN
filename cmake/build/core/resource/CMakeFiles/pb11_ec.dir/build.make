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
include core/resource/CMakeFiles/pb11_ec.dir/depend.make

# Include the progress variables for this target.
include core/resource/CMakeFiles/pb11_ec.dir/progress.make

# Include the compile flags for this target's objects.
include core/resource/CMakeFiles/pb11_ec.dir/flags.make

core/resource/CMakeFiles/pb11_ec.dir/pb11_ec.cc.o: core/resource/CMakeFiles/pb11_ec.dir/flags.make
core/resource/CMakeFiles/pb11_ec.dir/pb11_ec.cc.o: ../../core/resource/pb11_ec.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object core/resource/CMakeFiles/pb11_ec.dir/pb11_ec.cc.o"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/resource && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pb11_ec.dir/pb11_ec.cc.o -c /home/songzhen/workspace/ADGNN/core/resource/pb11_ec.cc

core/resource/CMakeFiles/pb11_ec.dir/pb11_ec.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pb11_ec.dir/pb11_ec.cc.i"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/resource && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/songzhen/workspace/ADGNN/core/resource/pb11_ec.cc > CMakeFiles/pb11_ec.dir/pb11_ec.cc.i

core/resource/CMakeFiles/pb11_ec.dir/pb11_ec.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pb11_ec.dir/pb11_ec.cc.s"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/resource && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/songzhen/workspace/ADGNN/core/resource/pb11_ec.cc -o CMakeFiles/pb11_ec.dir/pb11_ec.cc.s

# Object files for target pb11_ec
pb11_ec_OBJECTS = \
"CMakeFiles/pb11_ec.dir/pb11_ec.cc.o"

# External object files for target pb11_ec
pb11_ec_EXTERNAL_OBJECTS =

lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/resource/CMakeFiles/pb11_ec.dir/pb11_ec.cc.o
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/resource/CMakeFiles/pb11_ec.dir/build.make
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: lib/libcpptest.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/store/libstore.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/service/libservice.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/data_preprocess/libdata_preprocess.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/graph_build/libgraphbuild.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/sample/libsample.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/graph_build/libgraphbuild.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/store/libstore.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/service/libservice.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/util/libutil.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/partition/libpartition.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/store/libstore.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/service/libservice.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/util/libutil.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/partition/libpartition.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/structure/libstructure.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: libhw_grpc_proto.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libgrpc++_reflection.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libgrpc++.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libgrpc.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libz.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libcares.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libaddress_sorting.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libre2.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_raw_hash_set.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_hashtablez_sampler.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_hash.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_city.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_low_level_hash.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_statusor.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_bad_variant_access.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libgpr.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libupb.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_status.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_random_distributions.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_random_seed_sequences.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_random_internal_pool_urbg.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_random_internal_randen.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_random_internal_randen_hwaes.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_random_internal_randen_hwaes_impl.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_random_internal_randen_slow.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_random_internal_platform.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_random_internal_seed_material.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_random_seed_gen_exception.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_cord.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_bad_optional_access.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_cordz_info.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_cord_internal.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_cordz_functions.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_exponential_biased.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_cordz_handle.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_str_format_internal.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_synchronization.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_stacktrace.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_symbolize.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_debugging_internal.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_demangle_internal.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_graphcycles_internal.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_malloc_internal.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_time.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_strings.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_throw_delegate.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_int128.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_strings_internal.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_base.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_spinlock_wait.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_raw_logging_internal.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_log_severity.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_civil_time.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libabsl_time_zone.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libssl.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libcrypto.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libprotobuf.a
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: /home/songzhen/anaconda3/envs/python3.6/lib/libpython3.6m.so
lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so: core/resource/CMakeFiles/pb11_ec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/songzhen/workspace/ADGNN/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ../../lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so"
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/resource && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pb11_ec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
core/resource/CMakeFiles/pb11_ec.dir/build: lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so

.PHONY : core/resource/CMakeFiles/pb11_ec.dir/build

core/resource/CMakeFiles/pb11_ec.dir/clean:
	cd /home/songzhen/workspace/ADGNN/cmake/build/core/resource && $(CMAKE_COMMAND) -P CMakeFiles/pb11_ec.dir/cmake_clean.cmake
.PHONY : core/resource/CMakeFiles/pb11_ec.dir/clean

core/resource/CMakeFiles/pb11_ec.dir/depend:
	cd /home/songzhen/workspace/ADGNN/cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/songzhen/workspace/ADGNN /home/songzhen/workspace/ADGNN/core/resource /home/songzhen/workspace/ADGNN/cmake/build /home/songzhen/workspace/ADGNN/cmake/build/core/resource /home/songzhen/workspace/ADGNN/cmake/build/core/resource/CMakeFiles/pb11_ec.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : core/resource/CMakeFiles/pb11_ec.dir/depend

