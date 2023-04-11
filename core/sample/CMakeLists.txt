aux_source_directory(. DIR_LIB_SRCS)
add_library(sample  ${DIR_LIB_SRCS})
target_link_libraries(sample graphbuild util)