cmake_minimum_required(VERSION 2.8)

PROJECT(GRAPHEMBEDDING)

find_package(MPI REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -Wall -std=c++11 -Wno-sign-compare -fno-omit-frame-pointer -fopenmp")

set(MULTIVERSO_INC ${MULTIVERSO_DIR}/include)
set(MULTIVERSO_LIB ${MULTIVERSO_DIR}/build/src)
set(MULTIVERSO_SRC ${MULTIVERSO_DIR}/src)

include_directories(${MULTIVERSO_INC})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(/opt/OpenBLAS/include)

link_directories(${MULTIVERSO_LIB})
link_directories(/opt/OpenBLAS/lib)

set(SRCDIR ${PROJECT_SOURCE_DIR}/src)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_ROOT)

set(SRC ${MULTIVERSO_SRC} ${SRC_ROOT})

add_executable(graphembedding ${SRC})

target_link_libraries(graphembedding multiverso ${MPI_CXX_LIBRARIES})
