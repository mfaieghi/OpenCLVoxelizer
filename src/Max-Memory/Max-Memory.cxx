#define VOXEL_SIZE 0.003

#define STL_FILE_NAME "..\\model\\bunny_100times.stl" 

#define DEVICE_VENDOR 1
#define DEVICE_TYPE CL_DEVICE_TYPE_CPU

//#define CREATE_VOXELS
//#define RENDER

//#define WRITE_FILE
//#define OUTPUT_FILE_NAME "voxelized.stl"

//#define SOLID_VOXELIZATION
//#define DOUBLE_ACTOR
//#define CROSS_SECTION_X dim_grid[0]
//#define CROSS_SECTION_Y dim_grid[1]
//#define CROSS_SECTION_Z dim_grid[2]

//#define OBJECT_ACTOR
//#define AXES_ACTOR
//#define OUTLINE_ACTOR

#define _CRT_SECURE_NO_WARNINGS
#define KERNEL_FILE "..\\VoxelizerKernel.cl"
#define HALF_VOXEL_SIZE VOXEL_SIZE/2.0f;


//C header files
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
//OpenCL header files
#include <CL/cl.h>
//VTK header files
#include <vtkAxesActor.h>
#include <vtkTransform.h>
#include <vtkPLYWriter.h>
#include <vtkSTLWriter.h>
#include "vtkProperty.h"
#include <vtkOBJReader.h>
#include <vtkCubeSource.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkInteractorStyleTrackball.h>
#include <vtkPolyData.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSetMapper.h>
#include "vtkVoxel.h"
#include "vtkCellArray.h"
#include "vtkTriangleFilter.h"
#include "vtkDataSetSurfaceFilter.h"
#include "vtkOutlineFilter.h"
#include <vtkOrientationMarkerWidget.h>


cl_device_id create_device() {

	cl_platform_id *platform;
	cl_device_id dev;
	cl_uint num_platform;
	int err;

	/* Identify a platform */
	err = clGetPlatformIDs(0, NULL, &num_platform);
	if (err < 0) {
		printf("Error code: %d. Couldn't identify a platform\n", err);
		exit(1);
	}
	platform = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platform);
	clGetPlatformIDs(num_platform, platform, NULL);
	/* Access a device */
	err = clGetDeviceIDs(platform[DEVICE_VENDOR], DEVICE_TYPE, 1, &dev, NULL);

	if (err < 0) {
		printf("Error code: %d. Couldn't access any devices\n", err);
		exit(1);
	}

	return dev;
}


cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	int err;

	/* Read program file and place content into buffer */
	program_handle = fopen(filename, "r");
	if (program_handle == NULL) {
		printf("Couldn't find the program file\n");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1,
		(const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create the program\n", err);
		exit(1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(
		program,
		0,
		NULL,
		NULL,
		NULL,
		NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	return program;
}


void print_device_info(cl_device_id dev){

	cl_ulong glob_mem_size, local_mem_size, allocation_mem_size;
	cl_uint clock_freq, num_core, work_item_dim, time_res;
	size_t local_size, work_item_size[3];
	char dev_vendor[40], dev_name[400], driver_version[40], device_version[40];

	clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(dev_vendor), &dev_vendor, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dev_name), &dev_name, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(glob_mem_size), &glob_mem_size, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
	clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(driver_version), &driver_version, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(device_version), &device_version, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_freq), &clock_freq, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_core), &num_core, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(work_item_size), &work_item_size, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(work_item_dim), &work_item_dim, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(time_res), &time_res, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(allocation_mem_size), &allocation_mem_size, NULL);

	printf("==========================================================\n");
	printf("Device Sepc without consideration of kernels:\n");
	printf("CL_DEVICE_VENDOR:                     %s\n", dev_vendor);
	printf("CL_DEVICE_NAME:                       %s\n", dev_name);
	printf("CL_DEVICE_GLOBAL_MEM_SIZE:            %f GB\n", (float)glob_mem_size / (float)1073741824);
	printf("CL_DEVICE_LOCAL_MEM_SIZE:             %u KB\n", local_mem_size / 1024);
	printf("CL_DRIVER_VERSION:                    %s\n", driver_version);
	printf("CL_DEVICE_VERSION:                    %s\n", device_version);
	printf("CL_DEVICE_MAX_CLOCK_FREQUENCY:        %u MHz\n", clock_freq);
	printf("CL_DEVICE_MAX_COMPUTE_UNITS:          %u\n", num_core);
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE         %u\n", local_size);
	printf("CL_DEVICE_MAX_WORK_ITEM_SIZES:        {%u, %u, %u}\n", work_item_size[0], work_item_size[1], work_item_size[2]);
	printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:   %u\n", work_item_dim);
	printf("CL_DEVICE_PROFILING_TIMER_RESOLUTION: %u ns\n", (int)time_res);
	printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE:         %f GB\n", (float)allocation_mem_size / (float)1073741824);
	printf("==========================================================\n");

}


void make_voxel(
	vtkSmartPointer <vtkUnstructuredGrid> ug,
	vtkSmartPointer<vtkPoints> points,
	float coords[3],
	int counter) {

	vtkSmartPointer<vtkVoxel> voxel =
		vtkSmartPointer<vtkVoxel>::New();
	float x = coords[0];
	float y = coords[1];
	float z = coords[2];

	for (int i = 0; i < 8; ++i)
	{
		voxel->GetPointIds()->SetId(i, i + counter * 8);
	}
	ug->InsertNextCell(voxel->GetCellType(), voxel->GetPointIds());
	points->InsertNextPoint(x, y, z);
	points->InsertNextPoint(x + VOXEL_SIZE, y, z);
	points->InsertNextPoint(x, y + VOXEL_SIZE, z);
	points->InsertNextPoint(x + VOXEL_SIZE, y + VOXEL_SIZE, z);
	points->InsertNextPoint(x, y, z + VOXEL_SIZE);
	points->InsertNextPoint(x + VOXEL_SIZE, y, z + VOXEL_SIZE);
	points->InsertNextPoint(x, y + VOXEL_SIZE, z + VOXEL_SIZE);
	points->InsertNextPoint(x + VOXEL_SIZE, y + VOXEL_SIZE, z + VOXEL_SIZE);

}


int main()
{
	//Initialization
	cl_device_id device = create_device();
	print_device_info(device);
	cl_int err;
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create a context\n", err);
		exit(1);
	}
	cl_program program = build_program(context, device, KERNEL_FILE);
	cl_command_queue queue = clCreateCommandQueue(context, device,
		CL_QUEUE_PROFILING_ENABLE, &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create a command queue\n", err);
		exit(1);
	};

	//Read from file
	std::string inputFilename = STL_FILE_NAME;
	vtkSmartPointer<vtkSTLReader> reader =
		vtkSmartPointer<vtkSTLReader>::New();
	reader->SetFileName(inputFilename.c_str());
	reader->Update();
	vtkSmartPointer<vtkPolyData> mesh = reader->GetOutput();
	vtkSmartPointer<vtkPoints> points = mesh->GetPoints();
	vtkSmartPointer<vtkDataArray> dataArray = points->GetData();

	//Input data processing
	vtkIdType triangles_count = mesh->GetNumberOfCells();
	vtkSmartPointer<vtkIdList> faceIndex =
		vtkSmartPointer<vtkIdList>::New();
	vtkIdType vertexIndex = 0;
	float *coords = (float*)malloc(sizeof(float)* triangles_count * 9);
	for (vtkIdType i = 0; i < triangles_count; i++){
		int j = 9 * i;
		mesh->GetCellPoints(i, faceIndex);
		vertexIndex = faceIndex->GetId(0);
		coords[j] = dataArray->GetComponent(vertexIndex, 0);
		coords[j + 1] = dataArray->GetComponent(vertexIndex, 1);
		coords[j + 2] = dataArray->GetComponent(vertexIndex, 2);
		vertexIndex = faceIndex->GetId(1);
		coords[j + 3] = dataArray->GetComponent(vertexIndex, 0);
		coords[j + 4] = dataArray->GetComponent(vertexIndex, 1);
		coords[j + 5] = dataArray->GetComponent(vertexIndex, 2);
		vertexIndex = faceIndex->GetId(2);
		coords[j + 6] = dataArray->GetComponent(vertexIndex, 0);
		coords[j + 7] = dataArray->GetComponent(vertexIndex, 1);
		coords[j + 8] = dataArray->GetComponent(vertexIndex, 2);
	}

	//Create voxel grid
	double *boundsGeometry = mesh->GetBounds();
	float min_corner_object[3] = {
		boundsGeometry[0],
		boundsGeometry[2],
		boundsGeometry[4]
	};
	float max_corner_object[3] = {
		boundsGeometry[1],
		boundsGeometry[3],
		boundsGeometry[5]
	};
	unsigned int dim_grid[3] = {
		ceil((max_corner_object[0] - min_corner_object[0]) / VOXEL_SIZE),
		ceil((max_corner_object[1] - min_corner_object[1]) / VOXEL_SIZE),
		ceil((max_corner_object[2] - min_corner_object[2]) / VOXEL_SIZE)
	};
	if (dim_grid[0] == 0) dim_grid[0] = 1;
	if (dim_grid[1] == 0) dim_grid[1] = 1;
	if (dim_grid[2] == 0) dim_grid[2] = 1;
	float min_corner_grid[3] = {
		min_corner_object[0],
		min_corner_object[1],
		min_corner_object[2]
	};
	float max_corner_grid[3] = {
		min_corner_object[0] + dim_grid[0] * VOXEL_SIZE,
		min_corner_object[1] + dim_grid[1] * VOXEL_SIZE,
		min_corner_object[2] + dim_grid[2] * VOXEL_SIZE
	};
	cl_ulong voxel_counts = (cl_ulong)dim_grid[0] * (cl_ulong)dim_grid[1] * (cl_ulong)dim_grid[2]; //bits required
	cl_ulong array_length = voxel_counts / (cl_ulong)32; //ints required
	if (array_length % (cl_ulong)10 != 0) { array_length++;}
	float required_GB = array_length * (float)32/ (float)8589934592;
	float allowable_GB = 3.98;
	int allowable_ints = ceil(allowable_GB * (float)8589934592 / (float)32);
	//cl_ulong array_length_1 = (cl_ulong)allowable_ints;
	//cl_ulong array_length_2 = (cl_ulong)225000000;
	//cl_ulong array_length_3 = (cl_ulong)225000000;
	cl_ulong array_length_1 = allowable_ints;
	cl_ulong array_length_2 = allowable_ints;
	cl_ulong array_length_3 = allowable_ints;
	cl_ulong array_length_4 = array_length - (allowable_ints * 3);
	cl_uint *density_1 = (cl_uint*)malloc(sizeof(cl_uint)*array_length_1);
	cl_uint *density_2 = (cl_uint*)malloc(sizeof(cl_uint)*array_length_2);
	cl_uint *density_3 = (cl_uint*)malloc(sizeof(cl_uint)*array_length_3);
	cl_uint *density_4 = (cl_uint*)malloc(sizeof(cl_uint)*array_length_4);

	for (int i = 0; i < array_length_1; i++) { density_1[i] = 0;}
	for (int i = 0; i < array_length_2; i++) { density_2[i] = 0; }
	for (int i = 0; i < array_length_3; i++) { density_3[i] = 0; }
	for (int i = 0; i < array_length_4; i++) { density_4[i] = 0; }
	printf("Number of Triangles: %d\n", triangles_count);
	printf("Min and Max of Geometry: x_min = %f, y_min = %f, z_min = %f, \n x_max = %f, y_max = %f, z_max = %f\n",
		min_corner_object[0], min_corner_object[1], min_corner_object[2], max_corner_object[0], max_corner_object[1],
		max_corner_object[2]);
	printf("Voxel Size: %f, \n", VOXEL_SIZE);
	printf("Voxel Grid: %d, %d, %d\n", dim_grid[0], dim_grid[1], dim_grid[2]);
	printf("Voxel Counts: %I64u\n", voxel_counts);
	printf("Required GB: %f --> Required MB: %f", required_GB, required_GB * 1024);
	printf("Array Length: %I64u divided into %I64u and %I64u long arrays\n", array_length, array_length_1, array_length_2);
	/*Kernel configurations and execution*/
	size_t voxelization_local_size, voxelization_global_size, max_local_size, voxelization_revised_global_size;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_local_size), &max_local_size, NULL);
	voxelization_global_size = triangles_count;
	voxelization_local_size = max_local_size;
	if (voxelization_global_size % voxelization_local_size != 0) {
		voxelization_revised_global_size = (voxelization_global_size / voxelization_local_size + 1) * voxelization_local_size;
	}
	else {
		voxelization_revised_global_size = voxelization_global_size;
	}
	cl_mem min_corner_grid_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(float) * 3, min_corner_grid, &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create the h_minBoundsGrid_buffer\n", err);
		exit(1);
	};
	cl_mem dim_grid_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(int) * 3, dim_grid, &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create the dimGrid_buffer\n", err);
		exit(1);
	};
	cl_mem coords_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(float)* triangles_count * 9, coords, &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create the coords_buffer\n", err);
		exit(1);
	};
	cl_mem density_1_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * array_length_1, density_1, &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create the density_buffer_1\n", err);
		exit(1);
	};
	cl_mem density_2_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * array_length_2, density_2, &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create the density_buffer_2\n", err);
		exit(1);
	};
	cl_mem density_3_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * array_length_3, density_3, &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create the density_buffer_3\n", err);
		exit(1);
	};
	cl_mem density_4_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * array_length_4, density_4, &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create the density_buffer_4\n", err);
		exit(1);
	};
	cl_kernel voxelization_kernel;
	voxelization_kernel = clCreateKernel(program, "voxelizer", NULL);
	if (err < 0) {
		printf("Error code: %d. Couldn't create voxelization_kernel\n", err);
		exit(1);
	};
	err = clSetKernelArg(voxelization_kernel, 0, sizeof(cl_uint), &voxelization_global_size);
	float voxel_size = VOXEL_SIZE;
	err |= clSetKernelArg(voxelization_kernel, 1, sizeof(float), &voxel_size);
	err |= clSetKernelArg(voxelization_kernel, 2, sizeof(cl_mem), &min_corner_grid_buffer);
	err |= clSetKernelArg(voxelization_kernel, 3, sizeof(cl_mem), &dim_grid_buffer);
	err |= clSetKernelArg(voxelization_kernel, 4, sizeof(cl_mem), &coords_buffer);
	err |= clSetKernelArg(voxelization_kernel, 5, sizeof(cl_mem), &density_1_buffer);
	err |= clSetKernelArg(voxelization_kernel, 6, sizeof(cl_mem), &density_2_buffer);
	err |= clSetKernelArg(voxelization_kernel, 6, sizeof(cl_mem), &density_3_buffer);
	err |= clSetKernelArg(voxelization_kernel, 6, sizeof(cl_mem), &density_4_buffer);

	if (err < 0) {
		printf("Error code: %d. Couldn't create an argument for voxelization_kernel\n", err);
		exit(1);
	}
	
	//cl_ulong private_mem_size;
	//err = clGetKernelWorkGroupInfo(voxelization_kernel, device,
	//	CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(size_t), &private_mem_size, NULL);
	//printf("CL_KERNEL_PRIVATE_MEM_SIZE = %d", private_mem_size);
	
	cl_event voxelization_kernel_event;
	err = clEnqueueNDRangeKernel(queue, voxelization_kernel, 1, NULL, &voxelization_revised_global_size,
		&voxelization_local_size, 0, NULL, &voxelization_kernel_event);
	if (err < 0) {
		printf("Error code: %d. Couldn't enqueue the voxelization_kernel\n", err);
		exit(1);
	}
	err = clWaitForEvents(1, &voxelization_kernel_event);
	if (err < 0) {
		printf("Error code: %d, clWaitForEvent\n", err);
		exit(1);
	}
	size_t voxelization_kernel_time_start, voxelization_kernel_time_end, voxelization_kernel_elapsed_time;
	clGetEventProfilingInfo(voxelization_kernel_event, CL_PROFILING_COMMAND_START,
		sizeof(voxelization_kernel_time_start), &voxelization_kernel_time_start, NULL);
	clGetEventProfilingInfo(voxelization_kernel_event, CL_PROFILING_COMMAND_END,
		sizeof(voxelization_kernel_time_end), &voxelization_kernel_time_end, NULL);
	voxelization_kernel_elapsed_time = voxelization_kernel_time_end - voxelization_kernel_time_start;
	printf("Elapsed time in voxelization kernel: %f ms\n", (float)(voxelization_kernel_elapsed_time) / (float)1000000);

#ifdef SOLID_VOXELIZATION

	size_t x_ray_tracer_global_size[2] = { dim_grid[1], dim_grid[2] };
	size_t x_ray_tracer_local_size[2] = { 16, 16 };
	for (int i = 0; i < 2; i++) {
		if (x_ray_tracer_global_size[i] % x_ray_tracer_local_size[i] != 0) {
			x_ray_tracer_global_size[i] = 
				(x_ray_tracer_global_size[i] / x_ray_tracer_local_size[i] + 1) * x_ray_tracer_local_size[i];
		}
	}
	cl_kernel x_ray_tracer_kernel = clCreateKernel(program, "x_ray_tracer", &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create x_ray_tracer_kernel\n", err);
		exit(1);
	};
	err = clSetKernelArg(x_ray_tracer_kernel, 0, sizeof(cl_mem), &dim_grid_buffer);
	err |= clSetKernelArg(x_ray_tracer_kernel, 1, sizeof(cl_mem), &density_buffer);
	if (err < 0) {
		printf("Error code: %d. Couldn't create an argument for x_ray_tracer_kernel\n", err);
		exit(1);
	}
	cl_event x_ray_tracer_kernel_event;
	err = clEnqueueNDRangeKernel(queue, x_ray_tracer_kernel, 2, NULL, x_ray_tracer_global_size,
		x_ray_tracer_local_size, 0, NULL, &x_ray_tracer_kernel_event);
	if (err < 0) {
		printf("Error code: %d. Couldn't enqueue the x_ray_tracer_kernel\n", err);
		exit(1);
	}
	err = clWaitForEvents(1, &x_ray_tracer_kernel_event);
	if (err < 0) {
		printf("Error code: %d, first clWaitForEvent\n", err);
		exit(1);
	}
	size_t x_ray_tracer_kernel_time_start, x_ray_tracer_kernel_time_end, x_ray_tracer_kernel_elapsed_time;
	clGetEventProfilingInfo(x_ray_tracer_kernel_event, CL_PROFILING_COMMAND_START,
		sizeof(x_ray_tracer_kernel_time_start), &x_ray_tracer_kernel_time_start, NULL);
	clGetEventProfilingInfo(x_ray_tracer_kernel_event, CL_PROFILING_COMMAND_END,
		sizeof(x_ray_tracer_kernel_time_end), &x_ray_tracer_kernel_time_end, NULL);
	x_ray_tracer_kernel_elapsed_time = x_ray_tracer_kernel_time_end - x_ray_tracer_kernel_time_start;


	size_t y_ray_tracer_global_size[2] = { dim_grid[0], dim_grid[2] };
	size_t y_ray_tracer_local_size[2] = { 16, 16 };
	for (int i = 0; i < 2; i++) {
		if (y_ray_tracer_global_size[i] % y_ray_tracer_local_size[i] != 0) {
			y_ray_tracer_global_size[i] =
				(y_ray_tracer_global_size[i] / y_ray_tracer_local_size[i] + 1) * y_ray_tracer_local_size[i];
		}
	}
	cl_kernel y_ray_tracer_kernel = clCreateKernel(program, "y_ray_tracer", &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create y_ray_tracer_kernel\n", err);
		exit(1);
	};
	err = clSetKernelArg(y_ray_tracer_kernel, 0, sizeof(cl_mem), &dim_grid_buffer);
	err |= clSetKernelArg(y_ray_tracer_kernel, 1, sizeof(cl_mem), &density_buffer);
	if (err < 0) {
		printf("Error code: %d. Couldn't create an argument for y_ray_tracer_kernel\n", err);
		exit(1);
	}
	cl_event y_ray_tracer_kernel_event;
	err = clEnqueueNDRangeKernel(queue, y_ray_tracer_kernel, 2, NULL, y_ray_tracer_global_size,
		y_ray_tracer_local_size, 0, NULL, &y_ray_tracer_kernel_event);
	if (err < 0) {
		printf("Error code: %d. Couldn't enqueue the y_ray_tracer_kernel\n", err);
		exit(1);
	}
	err = clWaitForEvents(1, &y_ray_tracer_kernel_event);
	if (err < 0) {
		printf("Error code: %d, first clWaitForEvent\n", err);
		exit(1);
	}
	size_t y_ray_tracer_kernel_time_start, y_ray_tracer_kernel_time_end, y_ray_tracer_kernel_elapsed_time;
	clGetEventProfilingInfo(y_ray_tracer_kernel_event, CL_PROFILING_COMMAND_START,
		sizeof(y_ray_tracer_kernel_time_start), &y_ray_tracer_kernel_time_start, NULL);
	clGetEventProfilingInfo(x_ray_tracer_kernel_event, CL_PROFILING_COMMAND_END,
		sizeof(y_ray_tracer_kernel_time_end), &y_ray_tracer_kernel_time_end, NULL);
	y_ray_tracer_kernel_elapsed_time = y_ray_tracer_kernel_time_end - y_ray_tracer_kernel_time_start;
	
	
	size_t z_ray_tracer_global_size[2] = { dim_grid[0], dim_grid[1] };
	size_t z_ray_tracer_local_size[2] = { 16, 16 };
	for (int i = 0; i < 2; i++) {
		if (z_ray_tracer_global_size[i] % z_ray_tracer_local_size[i] != 0) {
			z_ray_tracer_global_size[i] = (z_ray_tracer_global_size[i] / z_ray_tracer_local_size[i] + 1) * z_ray_tracer_local_size[i];
		}
	}
	cl_kernel z_ray_tracer_kernel = clCreateKernel(program, "z_ray_tracer", &err);
	if (err < 0) {
		printf("Error code: %d. Couldn't create z_ray_tracer_kernel\n", err);
		exit(1);
	};
	err = clSetKernelArg(z_ray_tracer_kernel, 0, sizeof(cl_mem), &dim_grid_buffer);
	err |= clSetKernelArg(z_ray_tracer_kernel, 1, sizeof(cl_mem), &density_buffer);
	if (err < 0) {
		printf("Error code: %d. Couldn't create an argument for z_ray_tracer_kernel\n", err);
		exit(1);
	}
	cl_event z_ray_tracer_kernel_event;
	err = clEnqueueNDRangeKernel(queue, z_ray_tracer_kernel, 2, NULL, z_ray_tracer_global_size,
		z_ray_tracer_local_size, 0, NULL, &z_ray_tracer_kernel_event);
	if (err < 0) {
		printf("Error code: %d. Couldn't enqueue the z_ray_tracer_kernel\n", err);
		exit(1);
	}
	err = clWaitForEvents(1, &z_ray_tracer_kernel_event);
	if (err < 0) {
		printf("Error code: %d, first clWaitForEvent\n", err);
		exit(1);
	}
	size_t z_ray_tracer_kernel_time_start, z_ray_tracer_kernel_time_end, z_ray_tracer_kernel_elapsed_time;
	clGetEventProfilingInfo(z_ray_tracer_kernel_event, CL_PROFILING_COMMAND_START,
		sizeof(z_ray_tracer_kernel_time_start), &z_ray_tracer_kernel_time_start, NULL);
	clGetEventProfilingInfo(z_ray_tracer_kernel_event, CL_PROFILING_COMMAND_END,
		sizeof(z_ray_tracer_kernel_time_end), &z_ray_tracer_kernel_time_end, NULL);
	z_ray_tracer_kernel_elapsed_time = z_ray_tracer_kernel_time_end - z_ray_tracer_kernel_time_start;

	size_t temp = z_ray_tracer_kernel_time_end - x_ray_tracer_kernel_time_start;
	printf("Elpased time in solidization: %f ms\n", (float)temp / (float)1000000);
#endif


#ifdef CREATE_VOXELS
	// Retrieve kernel results
	void * density_1_mapped_memory = clEnqueueMapBuffer(queue, density_1_buffer, CL_TRUE,
		CL_MAP_READ, 0, sizeof(unsigned int) * array_length_1, 0, NULL, NULL, &err);
	if (err < 0) {
		printf("Error code : %d. Couldn't map the buffer to host memory\n", err);
		exit(1);
	}
	memcpy(density_1, density_1_mapped_memory, sizeof(unsigned int) * array_length_1);
	err = clEnqueueUnmapMemObject(queue, density_1_buffer, density_1_mapped_memory,
		0, NULL, NULL);
	if (err < 0) {
		printf("Error code: %d. Couldn't unmap the density_buffer\n", err);
		exit(1);
	}

	void * density_2_mapped_memory = clEnqueueMapBuffer(queue, density_2_buffer, CL_TRUE,
		CL_MAP_READ, 0, sizeof(unsigned int) * array_length_2, 0, NULL, NULL, &err);
	if (err < 0) {
		printf("Error code : %d. Couldn't map the buffer to host memory\n", err);
		exit(1);
	}
	memcpy(density_2, density_2_mapped_memory, sizeof(unsigned int) * array_length_2);
	err = clEnqueueUnmapMemObject(queue, density_2_buffer, density_2_mapped_memory,
		0, NULL, NULL);
	if (err < 0) {
		printf("Error code: %d. Couldn't unmap the density_buffer\n", err);
		exit(1);
	}

	//OpenCL clean up
	clReleaseDevice(device);
	clReleaseContext(context);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseKernel(voxelization_kernel);
	clReleaseMemObject(min_corner_grid_buffer);
	clReleaseMemObject(dim_grid_buffer);
	clReleaseMemObject(coords_buffer);
	clReleaseMemObject(density_1_buffer);
	clReleaseMemObject(density_2_buffer);
#ifdef SOLID_VOXELIZATION
	clReleaseKernel(x_ray_tracer_kernel);
	clReleaseKernel(y_ray_tracer_kernel);
	clReleaseKernel(z_ray_tracer_kernel);
#endif SOLID_VOXELIZATION


	/*Output data processing*/
#ifndef SOLID_VOXELIZATION
	vtkSmartPointer<vtkUnstructuredGrid> ug =
		vtkSmartPointer<vtkUnstructuredGrid>::New();
	vtkSmartPointer<vtkPoints> voxel_points =
		vtkSmartPointer<vtkPoints>::New();
	int filled_voxel_count = 0;

	for (cl_uint array_index = 0; array_index < array_length_1; array_index++) {

		for (int n = 0; n < 32; n++) {
			int mask = 1 << n;
			int masked_density = density_1[array_index] & mask;
			int the_bit = masked_density >> n;
			if (fabs(the_bit) == 1) {
				unsigned int voxel_index = 32 * array_index + n;
				int z = voxel_index / (dim_grid[0] * dim_grid[1]);
				int y = (voxel_index - z * dim_grid[0] * dim_grid[1]) / dim_grid[0];
				int x = voxel_index - z * dim_grid[0] * dim_grid[1] - y * dim_grid[0];
				float min_corner_voxel[3] = {
					min_corner_grid[0] + x * VOXEL_SIZE,
					min_corner_grid[1] + y * VOXEL_SIZE,
					min_corner_grid[2] + z * VOXEL_SIZE
				};
				make_voxel(ug, voxel_points, min_corner_voxel, filled_voxel_count);
				filled_voxel_count++;
			}
		}
		//printf("\n");
	}
	for (cl_uint array_index = 0; array_index < array_length_2; array_index++) {

		for (int n = 0; n < 32; n++) {
			int mask = 1 << n;
			int masked_density = density_2[array_index] & mask;
			int the_bit = masked_density >> n;
			if (fabs(the_bit) == 1) {
				unsigned int voxel_index = 32 * array_index + n;
				int z = voxel_index / (dim_grid[0] * dim_grid[1]);
				int y = (voxel_index - z * dim_grid[0] * dim_grid[1]) / dim_grid[0];
				int x = voxel_index - z * dim_grid[0] * dim_grid[1] - y * dim_grid[0];
				float min_corner_voxel[3] = {
					min_corner_grid[0] + x * VOXEL_SIZE,
					min_corner_grid[1] + y * VOXEL_SIZE,
					min_corner_grid[2] + z * VOXEL_SIZE
				};
				make_voxel(ug, voxel_points, min_corner_voxel, filled_voxel_count);
				filled_voxel_count++;
			}
		}
		//printf("\n");
	}
	ug->SetPoints(voxel_points);
	vtkSmartPointer<vtkDataSetMapper> voxelized_mapper =
		vtkSmartPointer<vtkDataSetMapper>::New();
	voxelized_mapper->SetInputData(ug);
	vtkSmartPointer<vtkActor> voxelized_actor =
		vtkSmartPointer<vtkActor>::New();
	voxelized_actor->SetMapper(voxelized_mapper);
	voxelized_actor->GetProperty()->SetColor(0, 0.5, 0);
	//voxelized_actor->GetProperty()->EdgeVisibilityOn();
	//voxelized_actor->GetProperty()->SetEdgeColor(1, 1, 1);

	printf("Filled voxel count: %d\n", filled_voxel_count);

#else
#ifdef DOUBLE_ACTOR

	vtkSmartPointer<vtkUnstructuredGrid> ug_surface =
		vtkSmartPointer<vtkUnstructuredGrid>::New();
	vtkSmartPointer<vtkUnstructuredGrid> ug_interior =
		vtkSmartPointer<vtkUnstructuredGrid>::New();
	vtkSmartPointer<vtkPoints> voxel_points_surface =
		vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkPoints> voxel_points_interior =
		vtkSmartPointer<vtkPoints>::New();
	int filled_voxel_count = 0;
	int filled_voxel_count_interior = 0;

	for (int voxel_index = 0; voxel_index < array_length; voxel_index++) {
		if (density[voxel_index] == 5) {

			int z = voxel_index / (dim_grid[0] * dim_grid[1]);
			int y = (voxel_index - z * dim_grid[0] * dim_grid[1]) / dim_grid[0];
			int x = voxel_index - z * dim_grid[0] * dim_grid[1] - y * dim_grid[0];
			if (x < CROSS_SECTION_X && y < CROSS_SECTION_Y && CROSS_SECTION_Z) {
				float min_corner_voxel[3] = {
					min_corner_grid[0] + x * VOXEL_SIZE,
					min_corner_grid[1] + y * VOXEL_SIZE,
					min_corner_grid[2] + z * VOXEL_SIZE
				};
				make_voxel(ug_surface, voxel_points_surface, min_corner_voxel, filled_voxel_count);
				filled_voxel_count++;
			}
		}
	}

	for (int voxel_index = 0; voxel_index < array_length; voxel_index++) {
		if (density[voxel_index] == 4) {

			int z = voxel_index / (dim_grid[0] * dim_grid[1]);
			int y = (voxel_index - z * dim_grid[0] * dim_grid[1]) / dim_grid[0];
			int x = voxel_index - z * dim_grid[0] * dim_grid[1] - y * dim_grid[0];
			if (x < CROSS_SECTION_X && y < CROSS_SECTION_Y && CROSS_SECTION_Z) {
				float min_corner_voxel[3] = {
					min_corner_grid[0] + x * VOXEL_SIZE,
					min_corner_grid[1] + y * VOXEL_SIZE,
					min_corner_grid[2] + z * VOXEL_SIZE
				};
				make_voxel(ug_interior, voxel_points_interior, min_corner_voxel, filled_voxel_count_interior);
				filled_voxel_count_interior++;
			}
		}
	}
	ug_surface->SetPoints(voxel_points_surface);
	vtkSmartPointer<vtkDataSetMapper> voxelized_mapper_surface =
		vtkSmartPointer<vtkDataSetMapper>::New();
	voxelized_mapper_surface->SetInputData(ug_surface);
	vtkSmartPointer<vtkActor> voxelized_actor_surface =
		vtkSmartPointer<vtkActor>::New();
	voxelized_actor_surface->SetMapper(voxelized_mapper_surface);
	voxelized_actor_surface->GetProperty()->SetColor(1, 0.4, 0);
	//voxelized_actor->GetProperty()->EdgeVisibilityOn();
	//voxelized_actor->GetProperty()->SetEdgeColor(1, 1, 1);
	ug_interior->SetPoints(voxel_points_interior);
	vtkSmartPointer<vtkDataSetMapper> voxelized_mapper_interior =
		vtkSmartPointer<vtkDataSetMapper>::New();
	voxelized_mapper_interior->SetInputData(ug_interior);
	vtkSmartPointer<vtkActor> voxelized_actor_interior =
		vtkSmartPointer<vtkActor>::New();
	voxelized_actor_interior->SetMapper(voxelized_mapper_interior);
	voxelized_actor_interior->GetProperty()->SetColor(0, 1, 0);
	//voxelized_actor->GetProperty()->EdgeVisibilityOn();
	//voxelized_actor->GetProperty()->SetEdgeColor(1, 1, 1);
	printf("Filled voxel count: %d\n", filled_voxel_count);
	printf("Filled voxel count interior: %d\n", filled_voxel_count_interior);
#else
vtkSmartPointer<vtkUnstructuredGrid> ug =
vtkSmartPointer<vtkUnstructuredGrid>::New();
vtkSmartPointer<vtkPoints> voxel_points =
vtkSmartPointer<vtkPoints>::New();
int filled_voxel_count = 0;

for (int voxel_index = 0; voxel_index < array_length; voxel_index++) {
	if (density[voxel_index] >= 4) {

		int z = voxel_index / (dim_grid[0] * dim_grid[1]);
		int y = (voxel_index - z * dim_grid[0] * dim_grid[1]) / dim_grid[0];
		int x = voxel_index - z * dim_grid[0] * dim_grid[1] - y * dim_grid[0];

		float min_corner_voxel[3] = {
			min_corner_grid[0] + x * VOXEL_SIZE,
			min_corner_grid[1] + y * VOXEL_SIZE,
			min_corner_grid[2] + z * VOXEL_SIZE
		};
		make_voxel(ug, voxel_points, min_corner_voxel, filled_voxel_count);
		filled_voxel_count++;
	}
}
ug->SetPoints(voxel_points);
vtkSmartPointer<vtkDataSetMapper> voxelized_mapper =
vtkSmartPointer<vtkDataSetMapper>::New();
voxelized_mapper->SetInputData(ug);
vtkSmartPointer<vtkActor> voxelized_actor =
vtkSmartPointer<vtkActor>::New();
voxelized_actor->SetMapper(voxelized_mapper);
voxelized_actor->GetProperty()->SetColor(1, 0.5, 0);
//voxelized_actor->GetProperty()->EdgeVisibilityOn();
//voxelized_actor->GetProperty()->SetEdgeColor(1, 1, 1);
printf("Filled voxel count: %d\n", filled_voxel_count);
#endif
#endif


#ifdef RENDER
#ifdef OBJECT_ACTOR
	vtkSmartPointer<vtkPolyDataMapper> object_mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
	object_mapper->SetInputConnection(reader->GetOutputPort());
	vtkSmartPointer<vtkActor> object_actor =
		vtkSmartPointer<vtkActor>::New();
	object_mapper->SetInputConnection(reader->GetOutputPort());
	object_actor->SetMapper(object_mapper);
	object_actor->GetProperty()->SetColor(1, 0.4, 0);
	object_actor->GetProperty()->EdgeVisibilityOn();
	//object_actor->GetProperty()->SetRepresentationToPoints();
	//object_actor->GetProperty()->SetRepresentationToSurface();
	//object_actor->GetProperty()->SetRepresentationToWireframe();
#endif
#ifdef AXES_ACTOR
	vtkSmartPointer<vtkAxesActor> axes_actor =
		vtkSmartPointer<vtkAxesActor>::New();
#endif
#ifdef OUTLINE_ACTOR
	vtkSmartPointer<vtkOutlineFilter> outline =
		vtkSmartPointer<vtkOutlineFilter>::New();
	outline->SetInputData(reader->GetOutput());
	vtkSmartPointer<vtkPolyDataMapper> outline_mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
	outline_mapper->SetInputConnection(outline->GetOutputPort());
	vtkSmartPointer<vtkActor> outline_actor =
		vtkSmartPointer<vtkActor>::New();
	outline_actor->SetMapper(outline_mapper);
	outline_actor->GetProperty()->SetColor(0,0,0);
#endif

	vtkSmartPointer<vtkRenderer> renderer =
		vtkSmartPointer<vtkRenderer>::New();
	renderer->SetBackground(1, 1, 1); // Background color green
#ifdef DOUBLE_ACTOR
	renderer->AddActor(voxelized_actor_surface);
	renderer->AddActor(voxelized_actor_interior);
#else
	renderer->AddActor(voxelized_actor);
#endif
#ifdef OBJECT_ACTOR
	renderer->AddActor(object_actor);
#endif
#ifdef AXES_ACTOR
	renderer->AddActor(axes_actor);
#endif
#ifdef OUTLINE_ACTOR
	renderer->AddActor(outline_actor);
#endif
	printf("over here\n");
	vtkSmartPointer<vtkRenderWindow> render_window =
		vtkSmartPointer<vtkRenderWindow>::New();
	render_window->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> render_window_interactor =
		vtkSmartPointer<vtkRenderWindowInteractor>::New();
	render_window_interactor->SetRenderWindow(render_window);
	render_window->Render();
	vtkSmartPointer<vtkInteractorStyleTrackballCamera> style =
		vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	render_window_interactor->SetInteractorStyle(style);
	render_window_interactor->Start();
#endif
#endif

#ifdef WRITE_FILE
	vtkSmartPointer<vtkDataSetSurfaceFilter> surface_filter =
		vtkSmartPointer<vtkDataSetSurfaceFilter>::New();
#ifdef SOLID_VOXELIZATION
#ifdef DOUBLE_ACTOR
	surface_filter->SetInputData(ug_surface);
#else
	surface_filter->SetInputData(ug);
#endif
#else
	surface_filter->SetInputData(ug);
#endif
	surface_filter->Update();
	surface_filter->Update();
	vtkSmartPointer<vtkTriangleFilter> triangle_filter =
		vtkSmartPointer<vtkTriangleFilter>::New();
	triangle_filter->SetInputConnection(surface_filter->GetOutputPort());
	triangle_filter->Update();
	vtkSmartPointer<vtkSTLWriter> stlWriter =
		vtkSmartPointer<vtkSTLWriter>::New();
	std::string name = "OUTPUT_FILE_NAME";
	stlWriter->SetFileName(name.c_str());
	stlWriter->SetInputConnection(triangle_filter->GetOutputPort());
	stlWriter->Write();
#endif

	return 0;
}