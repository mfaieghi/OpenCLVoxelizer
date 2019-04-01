A set of code for GPU accelerated voxelization of triangle mesh models.

Full description of the code can be found in this paper: https://www.tandfonline.com/doi/abs/10.1080/16864360.2018.1486961

OpenCL and VTK are required to use the code.

For OpenCL, download the OpenCL from the proper vendor:
AMD: http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/
Intel: https://software.intel.com/en-us/intel-opencl
NVIDIA: https://developer.nvidia.com/cuda-toolkit

For VTK, download the latest version form http://www.vtk.org/download/ and follow the following steps:

Download the VTK source code in a zip file and extract it in a folder named as VTK-Source.

Run CMake on VTK-Source and generate the VS project files in a new folder called VTK-Installation. Be sure about the following setting

•	BUILD_SHAREED_LIBS is checked.

•	BUILD_TESTING is unchecked.

•	CMAKE_CONFIGURATION_TYPES Debug;Release

•	CMAKE_INSTALL_PREFIX C:\Program Files\VTK (or C:\Program Files (x86)\VTK)

Once CMake generated the VS project files, build the solution for both Debug and Release configurations.

Next, create a new environment variable VTK_DIR whose value must be the address of VTK-Installation folder.

Also, Add the VTK-Installation/bin/Debug to the PATH variable.
