project                     (Nesoi)
cmake_minimum_required      (VERSION 3.1)

option                      (parallel           "Build Nesoi with parallelization"         ON)

# Default to Release
if                          (NOT CMAKE_BUILD_TYPE)
    set                     (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property            (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif                       (NOT CMAKE_BUILD_TYPE)

if                          (NOT parallel)
    add_definitions         (-DNESOI_NO_PARALLEL)
endif                       ()

include_directories         (include)

add_subdirectory            (bindings/python)

