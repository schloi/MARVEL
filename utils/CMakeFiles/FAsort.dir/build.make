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
CMAKE_BINARY_DIR = /home/schloi/devel/MARVEL_dev

# Include any dependencies generated for this target.
include utils/CMakeFiles/FAsort.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include utils/CMakeFiles/FAsort.dir/compiler_depend.make

# Include the progress variables for this target.
include utils/CMakeFiles/FAsort.dir/progress.make

# Include the compile flags for this target's objects.
include utils/CMakeFiles/FAsort.dir/flags.make

utils/CMakeFiles/FAsort.dir/__/lib/utils.c.o: utils/CMakeFiles/FAsort.dir/flags.make
utils/CMakeFiles/FAsort.dir/__/lib/utils.c.o: lib/utils.c
utils/CMakeFiles/FAsort.dir/__/lib/utils.c.o: utils/CMakeFiles/FAsort.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object utils/CMakeFiles/FAsort.dir/__/lib/utils.c.o"
	cd /home/schloi/devel/MARVEL_dev/utils && /usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT utils/CMakeFiles/FAsort.dir/__/lib/utils.c.o -MF CMakeFiles/FAsort.dir/__/lib/utils.c.o.d -o CMakeFiles/FAsort.dir/__/lib/utils.c.o -c /home/schloi/devel/MARVEL_dev/lib/utils.c

utils/CMakeFiles/FAsort.dir/__/lib/utils.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/FAsort.dir/__/lib/utils.c.i"
	cd /home/schloi/devel/MARVEL_dev/utils && /usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib/utils.c > CMakeFiles/FAsort.dir/__/lib/utils.c.i

utils/CMakeFiles/FAsort.dir/__/lib/utils.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/FAsort.dir/__/lib/utils.c.s"
	cd /home/schloi/devel/MARVEL_dev/utils && /usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib/utils.c -o CMakeFiles/FAsort.dir/__/lib/utils.c.s

utils/CMakeFiles/FAsort.dir/FAsort.c.o: utils/CMakeFiles/FAsort.dir/flags.make
utils/CMakeFiles/FAsort.dir/FAsort.c.o: utils/FAsort.c
utils/CMakeFiles/FAsort.dir/FAsort.c.o: utils/CMakeFiles/FAsort.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object utils/CMakeFiles/FAsort.dir/FAsort.c.o"
	cd /home/schloi/devel/MARVEL_dev/utils && /usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT utils/CMakeFiles/FAsort.dir/FAsort.c.o -MF CMakeFiles/FAsort.dir/FAsort.c.o.d -o CMakeFiles/FAsort.dir/FAsort.c.o -c /home/schloi/devel/MARVEL_dev/utils/FAsort.c

utils/CMakeFiles/FAsort.dir/FAsort.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/FAsort.dir/FAsort.c.i"
	cd /home/schloi/devel/MARVEL_dev/utils && /usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/utils/FAsort.c > CMakeFiles/FAsort.dir/FAsort.c.i

utils/CMakeFiles/FAsort.dir/FAsort.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/FAsort.dir/FAsort.c.s"
	cd /home/schloi/devel/MARVEL_dev/utils && /usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/utils/FAsort.c -o CMakeFiles/FAsort.dir/FAsort.c.s

# Object files for target FAsort
FAsort_OBJECTS = \
"CMakeFiles/FAsort.dir/__/lib/utils.c.o" \
"CMakeFiles/FAsort.dir/FAsort.c.o"

# External object files for target FAsort
FAsort_EXTERNAL_OBJECTS =

utils/FAsort: utils/CMakeFiles/FAsort.dir/__/lib/utils.c.o
utils/FAsort: utils/CMakeFiles/FAsort.dir/FAsort.c.o
utils/FAsort: utils/CMakeFiles/FAsort.dir/build.make
utils/FAsort: utils/CMakeFiles/FAsort.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/schloi/devel/MARVEL_dev/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable FAsort"
	cd /home/schloi/devel/MARVEL_dev/utils && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FAsort.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utils/CMakeFiles/FAsort.dir/build: utils/FAsort
.PHONY : utils/CMakeFiles/FAsort.dir/build

utils/CMakeFiles/FAsort.dir/clean:
	cd /home/schloi/devel/MARVEL_dev/utils && $(CMAKE_COMMAND) -P CMakeFiles/FAsort.dir/cmake_clean.cmake
.PHONY : utils/CMakeFiles/FAsort.dir/clean

utils/CMakeFiles/FAsort.dir/depend:
	cd /home/schloi/devel/MARVEL_dev && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/schloi/devel/MARVEL_dev /home/schloi/devel/MARVEL_dev/utils /home/schloi/devel/MARVEL_dev /home/schloi/devel/MARVEL_dev/utils /home/schloi/devel/MARVEL_dev/utils/CMakeFiles/FAsort.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utils/CMakeFiles/FAsort.dir/depend

