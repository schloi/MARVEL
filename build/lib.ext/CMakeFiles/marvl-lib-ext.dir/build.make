# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/schloi/devel/MARVEL_dev

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/schloi/devel/MARVEL_dev/build

# Include any dependencies generated for this target.
include lib.ext/CMakeFiles/marvl-lib-ext.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib.ext/CMakeFiles/marvl-lib-ext.dir/compiler_depend.make

# Include the progress variables for this target.
include lib.ext/CMakeFiles/marvl-lib-ext.dir/progress.make

# Include the compile flags for this target's objects.
include lib.ext/CMakeFiles/marvl-lib-ext.dir/flags.make

lib.ext/CMakeFiles/marvl-lib-ext.dir/bitarr.c.o: lib.ext/CMakeFiles/marvl-lib-ext.dir/flags.make
lib.ext/CMakeFiles/marvl-lib-ext.dir/bitarr.c.o: ../lib.ext/bitarr.c
lib.ext/CMakeFiles/marvl-lib-ext.dir/bitarr.c.o: lib.ext/CMakeFiles/marvl-lib-ext.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object lib.ext/CMakeFiles/marvl-lib-ext.dir/bitarr.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/lib.ext && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib.ext/CMakeFiles/marvl-lib-ext.dir/bitarr.c.o -MF CMakeFiles/marvl-lib-ext.dir/bitarr.c.o.d -o CMakeFiles/marvl-lib-ext.dir/bitarr.c.o -c /home/schloi/devel/MARVEL_dev/lib.ext/bitarr.c

lib.ext/CMakeFiles/marvl-lib-ext.dir/bitarr.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-lib-ext.dir/bitarr.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/lib.ext && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib.ext/bitarr.c > CMakeFiles/marvl-lib-ext.dir/bitarr.c.i

lib.ext/CMakeFiles/marvl-lib-ext.dir/bitarr.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-lib-ext.dir/bitarr.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/lib.ext && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib.ext/bitarr.c -o CMakeFiles/marvl-lib-ext.dir/bitarr.c.s

marvl-lib-ext: lib.ext/CMakeFiles/marvl-lib-ext.dir/bitarr.c.o
marvl-lib-ext: lib.ext/CMakeFiles/marvl-lib-ext.dir/build.make
.PHONY : marvl-lib-ext

# Rule to build all files generated by this target.
lib.ext/CMakeFiles/marvl-lib-ext.dir/build: marvl-lib-ext
.PHONY : lib.ext/CMakeFiles/marvl-lib-ext.dir/build

lib.ext/CMakeFiles/marvl-lib-ext.dir/clean:
	cd /home/schloi/devel/MARVEL_dev/build/lib.ext && $(CMAKE_COMMAND) -P CMakeFiles/marvl-lib-ext.dir/cmake_clean.cmake
.PHONY : lib.ext/CMakeFiles/marvl-lib-ext.dir/clean

lib.ext/CMakeFiles/marvl-lib-ext.dir/depend:
	cd /home/schloi/devel/MARVEL_dev/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/schloi/devel/MARVEL_dev /home/schloi/devel/MARVEL_dev/lib.ext /home/schloi/devel/MARVEL_dev/build /home/schloi/devel/MARVEL_dev/build/lib.ext /home/schloi/devel/MARVEL_dev/build/lib.ext/CMakeFiles/marvl-lib-ext.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib.ext/CMakeFiles/marvl-lib-ext.dir/depend
