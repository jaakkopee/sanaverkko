# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.28.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.28.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jaakkoprattala/Documents/koodii/sanaverkko

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jaakkoprattala/Documents/koodii/sanaverkko

# Include any dependencies generated for this target.
include CMakeFiles/_sanasyna.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/_sanasyna.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/_sanasyna.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/_sanasyna.dir/flags.make

CMakeFiles/_sanasyna.dir/sanasyna.cc.o: CMakeFiles/_sanasyna.dir/flags.make
CMakeFiles/_sanasyna.dir/sanasyna.cc.o: sanasyna.cc
CMakeFiles/_sanasyna.dir/sanasyna.cc.o: CMakeFiles/_sanasyna.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/jaakkoprattala/Documents/koodii/sanaverkko/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/_sanasyna.dir/sanasyna.cc.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/_sanasyna.dir/sanasyna.cc.o -MF CMakeFiles/_sanasyna.dir/sanasyna.cc.o.d -o CMakeFiles/_sanasyna.dir/sanasyna.cc.o -c /Users/jaakkoprattala/Documents/koodii/sanaverkko/sanasyna.cc

CMakeFiles/_sanasyna.dir/sanasyna.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/_sanasyna.dir/sanasyna.cc.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jaakkoprattala/Documents/koodii/sanaverkko/sanasyna.cc > CMakeFiles/_sanasyna.dir/sanasyna.cc.i

CMakeFiles/_sanasyna.dir/sanasyna.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/_sanasyna.dir/sanasyna.cc.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jaakkoprattala/Documents/koodii/sanaverkko/sanasyna.cc -o CMakeFiles/_sanasyna.dir/sanasyna.cc.s

# Object files for target _sanasyna
_sanasyna_OBJECTS = \
"CMakeFiles/_sanasyna.dir/sanasyna.cc.o"

# External object files for target _sanasyna
_sanasyna_EXTERNAL_OBJECTS =

_sanasyna.so: CMakeFiles/_sanasyna.dir/sanasyna.cc.o
_sanasyna.so: CMakeFiles/_sanasyna.dir/build.make
_sanasyna.so: /opt/homebrew/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib
_sanasyna.so: /Library/Frameworks/./sfml-audio.framework/Versions/2.6.1/sfml-audio
_sanasyna.so: /Library/Frameworks/./sfml-system.framework/Versions/2.6.1/sfml-system
_sanasyna.so: CMakeFiles/_sanasyna.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/jaakkoprattala/Documents/koodii/sanaverkko/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module _sanasyna.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/_sanasyna.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/_sanasyna.dir/build: _sanasyna.so
.PHONY : CMakeFiles/_sanasyna.dir/build

CMakeFiles/_sanasyna.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/_sanasyna.dir/cmake_clean.cmake
.PHONY : CMakeFiles/_sanasyna.dir/clean

CMakeFiles/_sanasyna.dir/depend:
	cd /Users/jaakkoprattala/Documents/koodii/sanaverkko && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jaakkoprattala/Documents/koodii/sanaverkko /Users/jaakkoprattala/Documents/koodii/sanaverkko /Users/jaakkoprattala/Documents/koodii/sanaverkko /Users/jaakkoprattala/Documents/koodii/sanaverkko /Users/jaakkoprattala/Documents/koodii/sanaverkko/CMakeFiles/_sanasyna.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/_sanasyna.dir/depend

