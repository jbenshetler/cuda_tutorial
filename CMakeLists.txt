cmake_minimum_required(VERSION 3.8 FATAL_ERROR)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_SOURCE_DIRECTORY ${CMAKE_PROJECT_DIR}/src)
include_directories(AFTER SYSTEM ${CMAKE_PROJECT_DIR}/include)

set(CMAKE_CXX_STANDARD 20)
project(cuda_tutorial LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

#set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "35;50;72")
 
#include(CTest)
 
# add_library(particles STATIC
#   randomize.cpp
#   randomize.h
#   particle.cu
#   particle.h
#   v3.cu
#   v3.h
#   )
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
#target_compile_features(hello PUBLIC cxx_std_11)
 
# set_target_properties( particles
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(hello src/hello.cu)
 
set_property(TARGET hello 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(particle_test PRIVATE particles)

add_executable(vector_add src/vector_add.cu)

