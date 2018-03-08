add_subdirectory            (pybind11)


if                          (CMAKE_LIBRARY_OUTPUT_DIRECTORY)        # to work with pip
    set                     (MODULE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
else                        ()
    set                     (MODULE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif                       ()

file                        (MAKE_DIRECTORY "${MODULE_OUTPUT_DIRECTORY}/nesoi")
file                        (GLOB NESOI_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/nesoi/*.py")
add_custom_target           (nesoi ALL ${CMAKE_COMMAND} -E copy_directory
                                       ${CMAKE_CURRENT_SOURCE_DIR}/nesoi ${MODULE_OUTPUT_DIRECTORY}/nesoi
                                       DEPENDS ${NESOI_PYTHON})

pybind11_add_module         (_nesoi nesoi.cpp tmt.cpp degree.cpp)
set_target_properties       (_nesoi PROPERTIES OUTPUT_NAME nesoi/_nesoi)
