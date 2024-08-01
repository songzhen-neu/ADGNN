aux_source_directory(. DIR_LIB_SRCS)
add_library(cpptest ${DIR_LIB_SRCS})
add_executable(pointer_test pointer_test.cc)
add_executable(address_or_value address_or_value.cc)
add_executable(vector_point_efficiency test_cpp/vector_point_efficiency.cpp)
add_executable(fundamental_test test_cpp/fundamental_test.cpp)
add_executable(bug test_cpp/bug.cpp)
add_executable(random test_cpp/random.cpp)
add_executable(random_eff test_cpp/random_eff.cpp)
add_executable(set_efficiency test_cpp/set_efficiency.cpp)
add_executable(get_length get_length.cc)
add_executable(array_variable_test array_variable_test.cc)
add_executable(vector_test vector_test.cc)
add_executable(map_insert_test map_insert_test.cc)
add_executable(MultiThread MultiThread.cc)
add_executable(staticTest staticTest.cc)
add_executable(test_condition_variable test_condition_variable.cc)
add_executable(test_wait test_wait.cc)
add_executable(test_matrix_multi test_matrix_multi.cc)
add_executable(test_vector_init testVectorInit.cpp)
add_executable(atomic_test atomic_test.cpp)
add_executable(testIterator testIterator.cpp)
add_executable(copy_test copy_test.cpp)
add_executable(test_mutex test_mutex.cc)
add_executable(multhread_efficiency test_multiple_threads/data_trans_efficiency.cpp)
add_executable(pass test_multiple_threads/address_value_pass.cpp)
add_executable(math_eff test_cpp/math_efficiency.cpp)


add_executable(hashset_test test_cpp/hashset_test.cpp)
target_link_libraries(hashset_test structure)


#add_subdirectory(test_system)
#target_link_libraries(testSetCtx service hw_grpc_proto)


#add_executable(startMaster startMaster.cc)
#add_executable(main_test main.cc)
find_package(Threads)



#target_link_libraries(startMaster client_lib dgnn_server_lib)
target_link_libraries(MultiThread ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(staticTest ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_condition_variable ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_wait ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_mutex ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(atomic_test ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(multhread_efficiency ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(main_test client_lib)


pybind11_add_module(datatrans test_pybind/DataTrans.cpp)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)