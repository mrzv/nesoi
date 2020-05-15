project                     (Nesoi)
cmake_minimum_required      (VERSION 3.1)

# C++11
set                         (CMAKE_CXX_STANDARD 11)

option                      (NESOI_PARALLEL           "Build Nesoi with parallelization"         ON)

# Default to Release
if                          (NOT CMAKE_BUILD_TYPE)
    set                     (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property            (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif                       (NOT CMAKE_BUILD_TYPE)

if                          (NOT NESOI_PARALLEL)
    add_definitions         (-DNESOI_NO_PARALLEL)
else                        ()
    find_package            (Threads)
    set                     (libraries ${libraries}     ${CMAKE_THREAD_LIBS_INIT})
endif                       ()

include_directories         (include)

add_subdirectory            (examples)
add_subdirectory            (bindings/python)

