include(FindPackageHandleStandardArgs)

# Check whether environment variable BLISS_DIR was set.
if(NOT BLISS_DIR)
   set(ENV_BLISS_DIR $ENV{BLISS_DIR})
   if(ENV_BLISS_DIR)
      set(BLISS_DIR $ENV{BLISS_DIR} CACHE PATH "Path to bliss build directory")
   endif()
endif()

# If the bliss directory is specified, first try to use it.
if(BLISS_DIR)
   set(COPY_BLISS_HEADERS FALSE)

   # Look for the includes with subdirectory bliss.
   find_path(BLISS_INCLUDE_DIR
      NAMES bliss/graph.hh
      PATHS ${BLISS_DIR}
      PATH_SUFFIXES include
      NO_DEFAULT_PATH
      )

   # If not found, look for the includes without bliss subdirectory.
   if(NOT BLISS_INCLUDE_DIR)
      find_path(BLISS_HEADER_DIR
         NAMES graph.hh
         PATHS ${BLISS_DIR}
         PATH_SUFFIXES include
         NO_DEFAULT_PATH
         )

      # If we found the headers there we copy the folder to a bliss folder in the binary dir and use that as include
      if(BLISS_HEADER_DIR)
         set(COPY_BLISS_HEADERS TRUE)
      endif()
   endif()

   # Look for the library in the bliss directory
   find_library(BLISS_LIBRARY
      NAMES bliss
      PATHS ${BLISS_DIR}
      PATH_SUFFIXES lib
      NO_DEFAULT_PATH
      )

   # If requested, copy the bliss headers to the <binary dir>/bliss/ and set include dir to <binary dir>.
   if(BLISS_LIBRARY AND COPY_BLISS_HEADERS)
      file(GLOB BLISS_HEADER_LIST ${BLISS_HEADER_DIR}/*.hh)
      file(COPY ${BLISS_HEADER_LIST} DESTINATION ${CMAKE_BINARY_DIR}/bliss)
      set(BLISS_INCLUDE_DIR ${CMAKE_BINARY_DIR} CACHE PATH "Include path for bliss headers" FORCE)
   endif()
endif()

# If bliss is not already found by the code above we look for it including system directories
if(NOT BLISS_INCLUDE_DIR OR NOT BLISS_LIBRARY)
   find_path(BLISS_INCLUDE_DIR
       NAMES bliss/graph.hh
       PATH_SUFFIXES include)

   find_library(BLISS_LIBRARY
       NAMES bliss
       PATH_SUFFIXES lib)
endif()

if(BLISS_INCLUDE_DIR AND BLISS_LIBRARY)
   set(BLISS_INCLUDE_DIRS ${BLISS_INCLUDE_DIR})
   set(BLISS_LIBRARIES ${BLISS_LIBRARY})

   # Check if bliss requires GMP.

   file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/check_bliss_uses_gmp.cpp" "#include <bliss/graph.hh>\n\nint main()\n{\n  bliss::Graph graph(0);\n  bliss::Stats stats;\n  graph.find_automorphisms(stats, NULL, NULL);\n  return 0;\n}\n")
   try_compile(COMPILE_RESULT "${CMAKE_CURRENT_BINARY_DIR}/" "${CMAKE_CURRENT_BINARY_DIR}/check_bliss_uses_gmp.cpp" LINK_LIBRARIES "${BLISS_LIBRARY}" CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${BLISS_INCLUDE_DIR}" OUTPUT_VARIABLE COMPILE_OUTPUT)

   if(NOT COMPILE_RESULT)
      # Bliss requires GMP.
      if(BLISS_FIND_REQUIRED)
         if(BLISS_FIND_QUIETLY)
            find_package(GMP REQUIRED QUIET)
         else()
            find_package(GMP REQUIRED)
         endif()
      else()
         if(BLISS_FIND_QUIETLY)
            find_package(GMP QUIET)
         else()
            find_package(GMP)
         endif()
      endif()
      if(GMP_FOUND)
         set(BLISS_INCLUDE_DIRS ${BLISS_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})
         set(BLISS_LIBRARIES ${BLISS_LIBRARIES} ${GMP_LIBRARIES})
         set(BLISS_DEFINITIONS "-DBLISS_USE_GMP" CACHE STRING "Extra CXX flags required for bliss")
         find_package_handle_standard_args(BLISS DEFAULT_MSG BLISS_INCLUDE_DIR BLISS_INCLUDE_DIRS BLISS_LIBRARIES BLISS_DEFINITIONS)
      elseif(NOT BLISS_FIND_QUIETLY)
         message(STATUS "Could NOT find BLISS (missing: GMP library)")
      endif()
   else()
      set(BLISS_DEFINITIONS " ")
      find_package_handle_standard_args(BLISS DEFAULT_MSG BLISS_INCLUDE_DIR BLISS_INCLUDE_DIRS BLISS_LIBRARIES BLISS_DEFINITIONS)
   endif()
   file(REMOVE "check_bliss_uses_gmp.cpp")
else()
   find_package_handle_standard_args(BLISS DEFAULT_MSG BLISS_INCLUDE_DIR BLISS_INCLUDE_DIRS BLISS_LIBRARIES BLISS_DEFINITIONS)
endif()

