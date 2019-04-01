#define HALF_VOXEL_SIZE voxel_size/2

int planeBoxOverlap(float voxel_size, float3 normal, float3 vert) {	// -NJMP-

	int q;
	float3 maxbox = (float3)(HALF_VOXEL_SIZE , HALF_VOXEL_SIZE , HALF_VOXEL_SIZE);

	float3 vmin, vmax;
	float v;
	
	//q=0
		v = vert.x;					// -NJMP-
		if (normal.x > 0.0f) {
			vmin.x = -maxbox.x - v;	// -NJMP-
			vmax.x = maxbox.x - v;	// -NJMP-
		}
		else {
			vmin.x = maxbox.x - v;	// -NJMP-
			vmax.x = -maxbox.x - v;	// -NJMP-
		}
	//q=1
		v = vert.y;					// -NJMP-
		if (normal.y > 0.0f) {
			vmin.y = -maxbox.y - v;	// -NJMP-
			vmax.y = maxbox.y - v;	// -NJMP-
		}
		else {
			vmin.y = maxbox.y - v;	// -NJMP-
			vmax.y = -maxbox.y - v;	// -NJMP-
		}

	//q=2
		v = vert.z;					// -NJMP-
		if (normal.z > 0.0f) {
			vmin.z = -maxbox.z - v;	// -NJMP-
			vmax.z = maxbox.z - v;	// -NJMP-
		}
		else {
			vmin.z = maxbox.z - v;	// -NJMP-
			vmax.z = -maxbox.z - v;	// -NJMP-
		}


	if (dot(normal, vmin) > 0.0f) return 0;	// -NJMP-
	if (dot(normal, vmax) >= 0.0f) return 1;	// -NJMP-
	return 0;
}


#define FINDMINMAX(x0,x1,x2,min,max) \
	min = max = x0;   \
if (x1<min) min = x1; \
if (x1>max) max = x1; \
if (x2<min) min = x2; \
if (x2>max) max = x2;

/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)			   \
	p0 = a*v0.y - b*v0.z;			       	   \
	p2 = a*v2.y - b*v2.z;			       	   \
if (p0<p2) { min = p0; max = p2; }\
else { min = p2; max = p0; } \
	rad = fa * HALF_VOXEL_SIZE + fb * HALF_VOXEL_SIZE;   \
if (min>rad || max<-rad) return 0;

#define AXISTEST_X2(a, b, fa, fb)			   \
	p0 = a*v0.y - b*v0.z;			           \
	p1 = a*v1.y - b*v1.z;			       	   \
if (p0<p1) { min = p0; max = p1; }\
else { min = p1; max = p0; } \
	rad = fa * HALF_VOXEL_SIZE + fb * HALF_VOXEL_SIZE;   \
if (min>rad || max<-rad) return 0;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)			   \
	p0 = -a*v0.x + b*v0.z;		      	   \
	p2 = -a*v2.x + b*v2.z;	       	       	   \
if (p0<p2) { min = p0; max = p2; }\
else { min = p2; max = p0; } \
	rad = fa * HALF_VOXEL_SIZE + fb * HALF_VOXEL_SIZE;   \
if (min>rad || max<-rad) return 0;

#define AXISTEST_Y1(a, b, fa, fb)			   \
	p0 = -a*v0.x + b*v0.z;		      	   \
	p1 = -a*v1.x + b*v1.z;	     	       	   \
if (p0<p1) { min = p0; max = p1; }\
else { min = p1; max = p0; } \
	rad = fa * HALF_VOXEL_SIZE + fb * HALF_VOXEL_SIZE;   \
if (min>rad || max<-rad) return 0;

/*======================== Z-tests ========================*/
#define AXISTEST_Z12(a, b, fa, fb)			   \
	p1 = a*v1.x - b*v1.y;			           \
	p2 = a*v2.x - b*v2.y;			       	   \
if (p2<p1) { min = p2; max = p1; }\
else { min = p1; max = p2; } \
	rad = fa * HALF_VOXEL_SIZE + fb * HALF_VOXEL_SIZE;   \
if (min>rad || max<-rad) return 0;

#define AXISTEST_Z0(a, b, fa, fb)			   \
	p0 = a*v0.x - b*v0.y;				   \
	p1 = a*v1.x - b*v1.y;			           \
if (p0<p1) { min = p0; max = p1; }\
else { min = p1; max = p0; } \
	rad = fa * HALF_VOXEL_SIZE + fb * HALF_VOXEL_SIZE;   \
if (min>rad || max<-rad) return 0;


int triBoxOverlap(float voxel_size, float3 center, float3 u0, float3 u1, float3 u2){
	
	float min, max, p0, p1, p2, rad, fex, fey, fez;
	float3 normal, e0, e1, e2;

	float3 v0 = u0 - center;
	float3 v1 = u1 - center;
	float3 v2 = u2 - center;

	e0 = v1 - v0;
	e1 = v2 - v1;
	e2 = v0 - v2;

	fex = fabs(e0.x);
	fey = fabs(e0.y);
	fez = fabs(e0.z);
	AXISTEST_X01(e0.z, e0.y, fez, fey);
	AXISTEST_Y02(e0.z, e0.x, fez, fex);
	AXISTEST_Z12(e0.y, e0.x, fey, fex);


	fex = fabs(e1.x);
	fey = fabs(e1.y);
	fez = fabs(e1.z);
	AXISTEST_X01(e1.z, e1.y, fez, fey);
	AXISTEST_Y02(e1.z, e1.x, fez, fex);
	AXISTEST_Z0(e1.y, e1.x, fey, fex);


	fex = fabs(e2.x);
	fey = fabs(e2.y);
	fez = fabs(e2.z);
	AXISTEST_X2(e2.z, e2.y, fez, fey);
	AXISTEST_Y1(e2.z, e2.x, fez, fex);
	AXISTEST_Z12(e2.y, e2.x, fey, fex);

	FINDMINMAX(v0.x, v1.x, v2.x, min, max);
	if (min>HALF_VOXEL_SIZE || max<-HALF_VOXEL_SIZE) return 0;
	FINDMINMAX(v0.y, v1.y, v2.y, min, max);
	if (min>HALF_VOXEL_SIZE || max<-HALF_VOXEL_SIZE) return 0;
	FINDMINMAX(v0.z, v1.z, v2.z, min, max);
	if (min>HALF_VOXEL_SIZE || max<-HALF_VOXEL_SIZE) return 0;


	normal = cross(e0, e1);
	if (!planeBoxOverlap(voxel_size, normal, v0)) return 0;


	return 1;
}



__kernel void voxelizer(int global_size,
	float h_voxel_size,
	__global float* h_min_corner_grid,
	__global int *h_dim_grid,
	__global float *coords,
	__global char *density) {

	//printf("local size is: %d\n", get_num_groups(0));
	int i = get_global_id(0);
	if (i < global_size) {

		__private float voxel_size = h_voxel_size;
		float3 min_corner_grid = (float3)(
			h_min_corner_grid[0],
			h_min_corner_grid[1],
			h_min_corner_grid[2]
			);
		int3 dim_grid = (int3)(
			h_dim_grid[0],
			h_dim_grid[1],
			h_dim_grid[2]
			);

		/*Triangle vertices*/
		__private float3 v0 = (float3)(coords[9 * i], coords[9 * i + 1], coords[9 * i + 2]);
		__private float3 v1 = (float3)(coords[9 * i + 3], coords[9 * i + 4], coords[9 * i + 5]);
		__private float3 v2 = (float3)(coords[9 * i + 6], coords[9 * i + 7], coords[9 * i + 8]);

		float3 min_corner_AABB = (float3)(
			fmin(v0.x, fmin(v1.x, v2.x)),
			fmin(v0.y, fmin(v1.y, v2.y)),
			fmin(v0.z, fmin(v1.z, v2.z))
			);
		float3 max_corner_AABB = (float3)(
			fmax(v0.x, fmax(v1.x, v2.x)),
			fmax(v0.y, fmax(v1.y, v2.y)),
			fmax(v0.z, fmax(v1.z, v2.z))
			);
		int3 min_corner_ID = (int3)(
			floor((min_corner_AABB.x - min_corner_grid.x) / voxel_size),
			floor((min_corner_AABB.y - min_corner_grid.y) / voxel_size),
			floor((min_corner_AABB.z - min_corner_grid.z) / voxel_size)
			);
		int3 max_corner_ID = (int3)(
			floor((max_corner_AABB.x - min_corner_grid.x) / voxel_size),
			floor((max_corner_AABB.y - min_corner_grid.y) / voxel_size),
			floor((max_corner_AABB.z - min_corner_grid.z) / voxel_size)
			);

		//printf("thread: %d --> min_corner_AABB: (%f, %f, %f), max_corner_AABB: (%f, %f, %f), min_corner_ID: (%d, %d, %d), max_corner_ID: (%d, %d, %d)\n",
		//	i, min_corner_AABB.x, min_corner_AABB.y, min_corner_AABB.z,
		//	max_corner_AABB.x, max_corner_AABB.y, max_corner_AABB.z,
		//	min_corner_ID.x, min_corner_ID.y, min_corner_ID.z,
		//	max_corner_ID.x, max_corner_ID.y, max_corner_ID.z);

		for (int j = min_corner_ID.z; j <= max_corner_ID.z; j++) {
			for (int k = min_corner_ID.y; k <= max_corner_ID.y; k++) {
				for (int l = min_corner_ID.x; l <= max_corner_ID.x; l++) {

						__private float3 center = (float3)(
							voxel_size / 2 + l * voxel_size + min_corner_grid.x,
							voxel_size / 2 + k * voxel_size + min_corner_grid.y,
							voxel_size / 2 + j * voxel_size + min_corner_grid.z
							);
						__private int voxel_index = l + k * dim_grid.x + j * dim_grid.x * dim_grid.y;
						//printf("in thread %d ==> for (l,k,j)=(%d, %d, %d)--> center = (%f, %f, %f), voxel_index = %d\n",
						//	i, l, k, j, center.x, center.y, center.z, voxel_index);
						if (density[voxel_index] != 5) {
							if (triBoxOverlap(voxel_size, center, v0, v1, v2)) {
								density[voxel_index] = 5;

							}
						}
				}
			}
		}
	}
}//End

__kernel void x_ray_tracer(
	__global int* dimGrid,
	__global char* density) {

	//printf("dimGrid: %d, %d, %d\n", dimGrid[0], dimGrid[1], dimGrid[2]);
	int yIndex = get_global_id(0);
	int zIndex = get_global_id(1);


	if (yIndex < dimGrid[1] && zIndex < dimGrid[2]) {

		//printf("yIndex: %d, zIndex: %d\n", yIndex, zIndex);


		int o = 0;
		while (o < dimGrid[0]) {
			if (density[o + yIndex * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1]] == 5) {
				break;
			}
			o++;
		}

		int p = dimGrid[0] - 1;
		while (p > 0) {

			if (density[p + yIndex * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1]] == 5) {
				break;
			}
			p--;
		}
		//printf("o: %d, p: %d\n", o, p);
		for (int q = o + 1; q <= p - 1; q++) {

			if (density[q + yIndex * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1]] != 5) {
				density[q + yIndex * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1]] = 2;
			}
		}
	}
}




__kernel void y_ray_tracer(
	__global int* dimGrid,
	__global char* density) {

	//printf("dimGrid: %d, %d, %d\n", dimGrid[0], dimGrid[1], dimGrid[2]);

	int xIndex = get_global_id(0);
	int zIndex = get_global_id(1);


	if (xIndex < dimGrid[0] && zIndex < dimGrid[2]) {

		//printf("yIndex: %d, zIndex: %d\n", yIndex, zIndex);


		int o = 0;
		while (o < dimGrid[1]) {
			if (density[xIndex + o * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1]] == 5) {
				break;
			}
			o++;
		}

		int p = dimGrid[1] - 1;
		while (p > 0) {

			if (density[xIndex + p * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1]] == 5) {
				break;
			}
			p--;
		}
		//printf("o: %d, p: %d\n", o, p);
		for (int q = o + 1; q <= p - 1; q++) {

			if (density[xIndex + q * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1]] != 5) {
				density[xIndex + q * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1]]++;
			}
		}
	}
}



__kernel void z_ray_tracer(
	__global int* dimGrid,
	__global char* density) {

	//printf("dimGrid: %d, %d, %d\n", dimGrid[0], dimGrid[1], dimGrid[2]);

	int xIndex = get_global_id(0);
	int yIndex = get_global_id(1);


	if (xIndex < dimGrid[0] && yIndex < dimGrid[2]) {

		//printf("yIndex: %d, zIndex: %d\n", yIndex, zIndex);


		int o = 0;
		while (o < dimGrid[2]) {
			if (density[xIndex + yIndex * dimGrid[0] + o * dimGrid[0] * dimGrid[1]] == 5) {
				break;
			}
			o++;
		}

		int p = dimGrid[2] - 1;
		while (p > 0) {

			if (density[xIndex + yIndex * dimGrid[0] + p * dimGrid[0] * dimGrid[1]] == 5) {
				break;
			}
			p--;
		}
		//printf("o: %d, p: %d\n", o, p);
		for (int q = o + 1; q <= p - 1; q++) {

			if (density[xIndex + yIndex * dimGrid[0] + q * dimGrid[0] * dimGrid[1]] != 5) {
				density[xIndex + yIndex * dimGrid[0] + q * dimGrid[0] * dimGrid[1]]++;
			}
		}
	}
}
//End of file