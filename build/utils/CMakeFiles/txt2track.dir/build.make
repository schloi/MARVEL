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
include utils/CMakeFiles/txt2track.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include utils/CMakeFiles/txt2track.dir/compiler_depend.make

# Include the progress variables for this target.
include utils/CMakeFiles/txt2track.dir/progress.make

# Include the compile flags for this target's objects.
include utils/CMakeFiles/txt2track.dir/flags.make

utils/CMakeFiles/txt2track.dir/txt2track.c.o: utils/CMakeFiles/txt2track.dir/flags.make
utils/CMakeFiles/txt2track.dir/txt2track.c.o: ../utils/txt2track.c
utils/CMakeFiles/txt2track.dir/txt2track.c.o: utils/CMakeFiles/txt2track.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object utils/CMakeFiles/txt2track.dir/txt2track.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/utils && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT utils/CMakeFiles/txt2track.dir/txt2track.c.o -MF CMakeFiles/txt2track.dir/txt2track.c.o.d -o CMakeFiles/txt2track.dir/txt2track.c.o -c /home/schloi/devel/MARVEL_dev/utils/txt2track.c

utils/CMakeFiles/txt2track.dir/txt2track.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/txt2track.dir/txt2track.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/utils && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/utils/txt2track.c > CMakeFiles/txt2track.dir/txt2track.c.i

utils/CMakeFiles/txt2track.dir/txt2track.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/txt2track.dir/txt2track.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/utils && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/utils/txt2track.c -o CMakeFiles/txt2track.dir/txt2track.c.s

# Object files for target txt2track
txt2track_OBJECTS = \
"CMakeFiles/txt2track.dir/txt2track.c.o"

# External object files for target txt2track
txt2track_EXTERNAL_OBJECTS = \
"/home/schloi/devel/MARVEL_dev/build/db/CMakeFiles/marvl-db.dir/DB.c.o" \
"/home/schloi/devel/MARVEL_dev/build/db/CMakeFiles/marvl-db.dir/QV.c.o" \
"/home/schloi/devel/MARVEL_dev/build/db/CMakeFiles/marvl-db.dir/FA2x.c.o" \
"/home/schloi/devel/MARVEL_dev/build/db/CMakeFiles/marvl-db.dir/fileUtils.c.o" \
"/home/schloi/devel/MARVEL_dev/build/lib/CMakeFiles/marvl-lib.dir/dmask.c.o" \
"/home/schloi/devel/MARVEL_dev/build/lib/CMakeFiles/marvl-lib.dir/oflags.c.o" \
"/home/schloi/devel/MARVEL_dev/build/lib/CMakeFiles/marvl-lib.dir/pass.c.o" \
"/home/schloi/devel/MARVEL_dev/build/lib/CMakeFiles/marvl-lib.dir/tracks.c.o" \
"/home/schloi/devel/MARVEL_dev/build/lib/CMakeFiles/marvl-lib.dir/compression.c.o" \
"/home/schloi/devel/MARVEL_dev/build/lib/CMakeFiles/marvl-lib.dir/read_loader.c.o" \
"/home/schloi/devel/MARVEL_dev/build/lib/CMakeFiles/marvl-lib.dir/trim.c.o" \
"/home/schloi/devel/MARVEL_dev/build/lib/CMakeFiles/marvl-lib.dir/utils.c.o" \
"/home/schloi/devel/MARVEL_dev/build/lib/CMakeFiles/marvl-lib.dir/iseparator.c.o"

utils/txt2track: utils/CMakeFiles/txt2track.dir/txt2track.c.o
utils/txt2track: db/CMakeFiles/marvl-db.dir/DB.c.o
utils/txt2track: db/CMakeFiles/marvl-db.dir/QV.c.o
utils/txt2track: db/CMakeFiles/marvl-db.dir/FA2x.c.o
utils/txt2track: db/CMakeFiles/marvl-db.dir/fileUtils.c.o
utils/txt2track: lib/CMakeFiles/marvl-lib.dir/dmask.c.o
utils/txt2track: lib/CMakeFiles/marvl-lib.dir/oflags.c.o
utils/txt2track: lib/CMakeFiles/marvl-lib.dir/pass.c.o
utils/txt2track: lib/CMakeFiles/marvl-lib.dir/tracks.c.o
utils/txt2track: lib/CMakeFiles/marvl-lib.dir/compression.c.o
utils/txt2track: lib/CMakeFiles/marvl-lib.dir/read_loader.c.o
utils/txt2track: lib/CMakeFiles/marvl-lib.dir/trim.c.o
utils/txt2track: lib/CMakeFiles/marvl-lib.dir/utils.c.o
utils/txt2track: lib/CMakeFiles/marvl-lib.dir/iseparator.c.o
utils/txt2track: utils/CMakeFiles/txt2track.dir/build.make
utils/txt2track: dalign/libmarvl-align.a
utils/txt2track: /usr/lib/x86_64-linux-gnu/libz.so
utils/txt2track: utils/CMakeFiles/txt2track.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable txt2track"
	cd /home/schloi/devel/MARVEL_dev/build/utils && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/txt2track.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utils/CMakeFiles/txt2track.dir/build: utils/txt2track
.PHONY : utils/CMakeFiles/txt2track.dir/build

utils/CMakeFiles/txt2track.dir/clean:
	cd /home/schloi/devel/MARVEL_dev/build/utils && $(CMAKE_COMMAND) -P CMakeFiles/txt2track.dir/cmake_clean.cmake
.PHONY : utils/CMakeFiles/txt2track.dir/clean

utils/CMakeFiles/txt2track.dir/depend:
	cd /home/schloi/devel/MARVEL_dev/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/schloi/devel/MARVEL_dev /home/schloi/devel/MARVEL_dev/utils /home/schloi/devel/MARVEL_dev/build /home/schloi/devel/MARVEL_dev/build/utils /home/schloi/devel/MARVEL_dev/build/utils/CMakeFiles/txt2track.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utils/CMakeFiles/txt2track.dir/depend

