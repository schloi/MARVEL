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
include dalign2/CMakeFiles/marvl-align2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include dalign2/CMakeFiles/marvl-align2.dir/compiler_depend.make

# Include the progress variables for this target.
include dalign2/CMakeFiles/marvl-align2.dir/progress.make

# Include the compile flags for this target's objects.
include dalign2/CMakeFiles/marvl-align2.dir/flags.make

dalign2/CMakeFiles/marvl-align2.dir/align.c.o: dalign2/CMakeFiles/marvl-align2.dir/flags.make
dalign2/CMakeFiles/marvl-align2.dir/align.c.o: ../dalign2/align.c
dalign2/CMakeFiles/marvl-align2.dir/align.c.o: dalign2/CMakeFiles/marvl-align2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object dalign2/CMakeFiles/marvl-align2.dir/align.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/dalign2 && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT dalign2/CMakeFiles/marvl-align2.dir/align.c.o -MF CMakeFiles/marvl-align2.dir/align.c.o.d -o CMakeFiles/marvl-align2.dir/align.c.o -c /home/schloi/devel/MARVEL_dev/dalign2/align.c

dalign2/CMakeFiles/marvl-align2.dir/align.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-align2.dir/align.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/dalign2 && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/dalign2/align.c > CMakeFiles/marvl-align2.dir/align.c.i

dalign2/CMakeFiles/marvl-align2.dir/align.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-align2.dir/align.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/dalign2 && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/dalign2/align.c -o CMakeFiles/marvl-align2.dir/align.c.s

# Object files for target marvl-align2
marvl__align2_OBJECTS = \
"CMakeFiles/marvl-align2.dir/align.c.o"

# External object files for target marvl-align2
marvl__align2_EXTERNAL_OBJECTS =

dalign2/libmarvl-align2.a: dalign2/CMakeFiles/marvl-align2.dir/align.c.o
dalign2/libmarvl-align2.a: dalign2/CMakeFiles/marvl-align2.dir/build.make
dalign2/libmarvl-align2.a: dalign2/CMakeFiles/marvl-align2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library libmarvl-align2.a"
	cd /home/schloi/devel/MARVEL_dev/build/dalign2 && $(CMAKE_COMMAND) -P CMakeFiles/marvl-align2.dir/cmake_clean_target.cmake
	cd /home/schloi/devel/MARVEL_dev/build/dalign2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/marvl-align2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dalign2/CMakeFiles/marvl-align2.dir/build: dalign2/libmarvl-align2.a
.PHONY : dalign2/CMakeFiles/marvl-align2.dir/build

dalign2/CMakeFiles/marvl-align2.dir/clean:
	cd /home/schloi/devel/MARVEL_dev/build/dalign2 && $(CMAKE_COMMAND) -P CMakeFiles/marvl-align2.dir/cmake_clean.cmake
.PHONY : dalign2/CMakeFiles/marvl-align2.dir/clean

dalign2/CMakeFiles/marvl-align2.dir/depend:
	cd /home/schloi/devel/MARVEL_dev/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/schloi/devel/MARVEL_dev /home/schloi/devel/MARVEL_dev/dalign2 /home/schloi/devel/MARVEL_dev/build /home/schloi/devel/MARVEL_dev/build/dalign2 /home/schloi/devel/MARVEL_dev/build/dalign2/CMakeFiles/marvl-align2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dalign2/CMakeFiles/marvl-align2.dir/depend
