cmake_minimum_required(VERSION 2.8)

project(Voxelization-OpenCL-VTK)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package( OpenCL REQUIRED )
find_package( VTK REQUIRED )

set(PROJECTS
Seidel
SAT
Bit-Masking
Cube-Based
Max-Memory
)


foreach(PROJECTS ${PROJECTS})
add_executable(${PROJECTS} src/${PROJECTS}/${PROJECTS}.cxx src/${PROJECTS}/${PROJECTS}.cl)
include(${VTK_USE_FILE})
include_directories(${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECTS} ${VTK_LIBRARIES} ${OpenCL_LIBRARY})
source_group(Kernel FILES src/${PROJECTS}/${PROJECTS}.cl)
endforeach(PROJECTS)