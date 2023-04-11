aux_source_directory(. DIR_LIB_SRCS)
add_library(data_preprocess ${DIR_LIB_SRCS})
add_executable(partition_graph partition_graph.cpp)
target_link_libraries(partition_graph util)