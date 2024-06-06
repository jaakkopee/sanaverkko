project(_sanasyna LANGUAGES CXX)

cmake_minimum_required(VERSION 3.25.1)


#sfml
find_package(SFML 2.5 COMPONENTS audio system REQUIRED)

#python
find_package(PythonLibs 3.6 REQUIRED)

# make python module
add_library(_sanasyna MODULE sanasyna.cc)
target_include_directories(_sanasyna PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(_sanasyna PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(_sanasyna PUBLIC ${SFML_INCLUDE_DIR})

target_link_libraries(_sanasyna PUBLIC sfml-audio sfml-system ${SFML_LIBRARIES})
target_link_libraries(_sanasyna PUBLIC ${PYTHON_LIBRARIES})
set_target_properties(_sanasyna PROPERTIES PREFIX "")
set_target_properties(_sanasyna PROPERTIES SUFFIX ".so")

find_package(pybind11 REQUIRED)
target_link_libraries(_sanasyna PUBLIC sfml-audio pybind11::module)
