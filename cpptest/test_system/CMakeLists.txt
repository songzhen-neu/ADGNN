aux_source_directory(. DIR_LIB_SRCS)
add_executable(testSetCtx testSetCtx.cc )
add_executable(test_set_embs test_set_embs.cc)

#target_link_libraries(test_set_embs PRIVATE bittest)
target_link_libraries(test_set_embs PRIVATE store)
target_link_libraries(test_set_embs PRIVATE service)




target_link_libraries(testSetCtx service)