aux_source_directory(. DIR_LIB_SRCS)
add_library(structure ${DIR_LIB_SRCS} Graph.cpp Graph.h SubGraph.cpp SubGraph.h GraphLayer.cpp GraphLayer.h)
target_link_libraries(structure pybind11::embed)
message(STATUS "structure: ${DIR_LIB_SRCS}")