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
include lib/CMakeFiles/marvl-lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/CMakeFiles/marvl-lib.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/CMakeFiles/marvl-lib.dir/progress.make

# Include the compile flags for this target's objects.
include lib/CMakeFiles/marvl-lib.dir/flags.make

lib/CMakeFiles/marvl-lib.dir/dmask.c.o: lib/CMakeFiles/marvl-lib.dir/flags.make
lib/CMakeFiles/marvl-lib.dir/dmask.c.o: ../lib/dmask.c
lib/CMakeFiles/marvl-lib.dir/dmask.c.o: lib/CMakeFiles/marvl-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object lib/CMakeFiles/marvl-lib.dir/dmask.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMakeFiles/marvl-lib.dir/dmask.c.o -MF CMakeFiles/marvl-lib.dir/dmask.c.o.d -o CMakeFiles/marvl-lib.dir/dmask.c.o -c /home/schloi/devel/MARVEL_dev/lib/dmask.c

lib/CMakeFiles/marvl-lib.dir/dmask.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-lib.dir/dmask.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib/dmask.c > CMakeFiles/marvl-lib.dir/dmask.c.i

lib/CMakeFiles/marvl-lib.dir/dmask.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-lib.dir/dmask.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib/dmask.c -o CMakeFiles/marvl-lib.dir/dmask.c.s

lib/CMakeFiles/marvl-lib.dir/oflags.c.o: lib/CMakeFiles/marvl-lib.dir/flags.make
lib/CMakeFiles/marvl-lib.dir/oflags.c.o: ../lib/oflags.c
lib/CMakeFiles/marvl-lib.dir/oflags.c.o: lib/CMakeFiles/marvl-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object lib/CMakeFiles/marvl-lib.dir/oflags.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMakeFiles/marvl-lib.dir/oflags.c.o -MF CMakeFiles/marvl-lib.dir/oflags.c.o.d -o CMakeFiles/marvl-lib.dir/oflags.c.o -c /home/schloi/devel/MARVEL_dev/lib/oflags.c

lib/CMakeFiles/marvl-lib.dir/oflags.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-lib.dir/oflags.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib/oflags.c > CMakeFiles/marvl-lib.dir/oflags.c.i

lib/CMakeFiles/marvl-lib.dir/oflags.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-lib.dir/oflags.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib/oflags.c -o CMakeFiles/marvl-lib.dir/oflags.c.s

lib/CMakeFiles/marvl-lib.dir/pass.c.o: lib/CMakeFiles/marvl-lib.dir/flags.make
lib/CMakeFiles/marvl-lib.dir/pass.c.o: ../lib/pass.c
lib/CMakeFiles/marvl-lib.dir/pass.c.o: lib/CMakeFiles/marvl-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object lib/CMakeFiles/marvl-lib.dir/pass.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMakeFiles/marvl-lib.dir/pass.c.o -MF CMakeFiles/marvl-lib.dir/pass.c.o.d -o CMakeFiles/marvl-lib.dir/pass.c.o -c /home/schloi/devel/MARVEL_dev/lib/pass.c

lib/CMakeFiles/marvl-lib.dir/pass.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-lib.dir/pass.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib/pass.c > CMakeFiles/marvl-lib.dir/pass.c.i

lib/CMakeFiles/marvl-lib.dir/pass.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-lib.dir/pass.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib/pass.c -o CMakeFiles/marvl-lib.dir/pass.c.s

lib/CMakeFiles/marvl-lib.dir/tracks.c.o: lib/CMakeFiles/marvl-lib.dir/flags.make
lib/CMakeFiles/marvl-lib.dir/tracks.c.o: ../lib/tracks.c
lib/CMakeFiles/marvl-lib.dir/tracks.c.o: lib/CMakeFiles/marvl-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object lib/CMakeFiles/marvl-lib.dir/tracks.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMakeFiles/marvl-lib.dir/tracks.c.o -MF CMakeFiles/marvl-lib.dir/tracks.c.o.d -o CMakeFiles/marvl-lib.dir/tracks.c.o -c /home/schloi/devel/MARVEL_dev/lib/tracks.c

lib/CMakeFiles/marvl-lib.dir/tracks.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-lib.dir/tracks.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib/tracks.c > CMakeFiles/marvl-lib.dir/tracks.c.i

lib/CMakeFiles/marvl-lib.dir/tracks.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-lib.dir/tracks.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib/tracks.c -o CMakeFiles/marvl-lib.dir/tracks.c.s

lib/CMakeFiles/marvl-lib.dir/compression.c.o: lib/CMakeFiles/marvl-lib.dir/flags.make
lib/CMakeFiles/marvl-lib.dir/compression.c.o: ../lib/compression.c
lib/CMakeFiles/marvl-lib.dir/compression.c.o: lib/CMakeFiles/marvl-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object lib/CMakeFiles/marvl-lib.dir/compression.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMakeFiles/marvl-lib.dir/compression.c.o -MF CMakeFiles/marvl-lib.dir/compression.c.o.d -o CMakeFiles/marvl-lib.dir/compression.c.o -c /home/schloi/devel/MARVEL_dev/lib/compression.c

lib/CMakeFiles/marvl-lib.dir/compression.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-lib.dir/compression.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib/compression.c > CMakeFiles/marvl-lib.dir/compression.c.i

lib/CMakeFiles/marvl-lib.dir/compression.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-lib.dir/compression.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib/compression.c -o CMakeFiles/marvl-lib.dir/compression.c.s

lib/CMakeFiles/marvl-lib.dir/read_loader.c.o: lib/CMakeFiles/marvl-lib.dir/flags.make
lib/CMakeFiles/marvl-lib.dir/read_loader.c.o: ../lib/read_loader.c
lib/CMakeFiles/marvl-lib.dir/read_loader.c.o: lib/CMakeFiles/marvl-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object lib/CMakeFiles/marvl-lib.dir/read_loader.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMakeFiles/marvl-lib.dir/read_loader.c.o -MF CMakeFiles/marvl-lib.dir/read_loader.c.o.d -o CMakeFiles/marvl-lib.dir/read_loader.c.o -c /home/schloi/devel/MARVEL_dev/lib/read_loader.c

lib/CMakeFiles/marvl-lib.dir/read_loader.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-lib.dir/read_loader.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib/read_loader.c > CMakeFiles/marvl-lib.dir/read_loader.c.i

lib/CMakeFiles/marvl-lib.dir/read_loader.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-lib.dir/read_loader.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib/read_loader.c -o CMakeFiles/marvl-lib.dir/read_loader.c.s

lib/CMakeFiles/marvl-lib.dir/trim.c.o: lib/CMakeFiles/marvl-lib.dir/flags.make
lib/CMakeFiles/marvl-lib.dir/trim.c.o: ../lib/trim.c
lib/CMakeFiles/marvl-lib.dir/trim.c.o: lib/CMakeFiles/marvl-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object lib/CMakeFiles/marvl-lib.dir/trim.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMakeFiles/marvl-lib.dir/trim.c.o -MF CMakeFiles/marvl-lib.dir/trim.c.o.d -o CMakeFiles/marvl-lib.dir/trim.c.o -c /home/schloi/devel/MARVEL_dev/lib/trim.c

lib/CMakeFiles/marvl-lib.dir/trim.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-lib.dir/trim.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib/trim.c > CMakeFiles/marvl-lib.dir/trim.c.i

lib/CMakeFiles/marvl-lib.dir/trim.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-lib.dir/trim.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib/trim.c -o CMakeFiles/marvl-lib.dir/trim.c.s

lib/CMakeFiles/marvl-lib.dir/utils.c.o: lib/CMakeFiles/marvl-lib.dir/flags.make
lib/CMakeFiles/marvl-lib.dir/utils.c.o: ../lib/utils.c
lib/CMakeFiles/marvl-lib.dir/utils.c.o: lib/CMakeFiles/marvl-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object lib/CMakeFiles/marvl-lib.dir/utils.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMakeFiles/marvl-lib.dir/utils.c.o -MF CMakeFiles/marvl-lib.dir/utils.c.o.d -o CMakeFiles/marvl-lib.dir/utils.c.o -c /home/schloi/devel/MARVEL_dev/lib/utils.c

lib/CMakeFiles/marvl-lib.dir/utils.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-lib.dir/utils.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib/utils.c > CMakeFiles/marvl-lib.dir/utils.c.i

lib/CMakeFiles/marvl-lib.dir/utils.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-lib.dir/utils.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib/utils.c -o CMakeFiles/marvl-lib.dir/utils.c.s

lib/CMakeFiles/marvl-lib.dir/iseparator.c.o: lib/CMakeFiles/marvl-lib.dir/flags.make
lib/CMakeFiles/marvl-lib.dir/iseparator.c.o: ../lib/iseparator.c
lib/CMakeFiles/marvl-lib.dir/iseparator.c.o: lib/CMakeFiles/marvl-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/schloi/devel/MARVEL_dev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object lib/CMakeFiles/marvl-lib.dir/iseparator.c.o"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMakeFiles/marvl-lib.dir/iseparator.c.o -MF CMakeFiles/marvl-lib.dir/iseparator.c.o.d -o CMakeFiles/marvl-lib.dir/iseparator.c.o -c /home/schloi/devel/MARVEL_dev/lib/iseparator.c

lib/CMakeFiles/marvl-lib.dir/iseparator.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/marvl-lib.dir/iseparator.c.i"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/schloi/devel/MARVEL_dev/lib/iseparator.c > CMakeFiles/marvl-lib.dir/iseparator.c.i

lib/CMakeFiles/marvl-lib.dir/iseparator.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/marvl-lib.dir/iseparator.c.s"
	cd /home/schloi/devel/MARVEL_dev/build/lib && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/schloi/devel/MARVEL_dev/lib/iseparator.c -o CMakeFiles/marvl-lib.dir/iseparator.c.s

marvl-lib: lib/CMakeFiles/marvl-lib.dir/dmask.c.o
marvl-lib: lib/CMakeFiles/marvl-lib.dir/oflags.c.o
marvl-lib: lib/CMakeFiles/marvl-lib.dir/pass.c.o
marvl-lib: lib/CMakeFiles/marvl-lib.dir/tracks.c.o
marvl-lib: lib/CMakeFiles/marvl-lib.dir/compression.c.o
marvl-lib: lib/CMakeFiles/marvl-lib.dir/read_loader.c.o
marvl-lib: lib/CMakeFiles/marvl-lib.dir/trim.c.o
marvl-lib: lib/CMakeFiles/marvl-lib.dir/utils.c.o
marvl-lib: lib/CMakeFiles/marvl-lib.dir/iseparator.c.o
marvl-lib: lib/CMakeFiles/marvl-lib.dir/build.make
.PHONY : marvl-lib

# Rule to build all files generated by this target.
lib/CMakeFiles/marvl-lib.dir/build: marvl-lib
.PHONY : lib/CMakeFiles/marvl-lib.dir/build

lib/CMakeFiles/marvl-lib.dir/clean:
	cd /home/schloi/devel/MARVEL_dev/build/lib && $(CMAKE_COMMAND) -P CMakeFiles/marvl-lib.dir/cmake_clean.cmake
.PHONY : lib/CMakeFiles/marvl-lib.dir/clean

lib/CMakeFiles/marvl-lib.dir/depend:
	cd /home/schloi/devel/MARVEL_dev/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/schloi/devel/MARVEL_dev /home/schloi/devel/MARVEL_dev/lib /home/schloi/devel/MARVEL_dev/build /home/schloi/devel/MARVEL_dev/build/lib /home/schloi/devel/MARVEL_dev/build/lib/CMakeFiles/marvl-lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/CMakeFiles/marvl-lib.dir/depend
