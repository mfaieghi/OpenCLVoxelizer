__kernel void voxelizer
(
	int global_size
	, float h_voxel_size
	, __global float * h_min_corner_grid
	, __global int * h_dim_grid
	, __global float * coords
	, __global char * density)
{
	int i = get_global_id(0);
	if (i < global_size)
	{
		__private float voxel_size = h_voxel_size;
		float3 min_corner_grid = { h_min_corner_grid[0], h_min_corner_grid[1], h_min_corner_grid[2] };
		int3 dim_grid = { h_dim_grid[0], h_dim_grid[1], h_dim_grid[2] };
		__private float3 v0 = { coords[9 * i], coords[9 * i + 1], coords[9 * i + 2] };
		__private float3 v1 = { coords[9 * i + 3], coords[9 * i + 4], coords[9 * i + 5] };
		__private float3 v2 = { coords[9 * i + 6], coords[9 * i + 7], coords[9 * i + 8] };

		float3 min_corner_AABB = fmin(v0, fmin(v1, v2));
		float3 max_corner_AABB = fmax(v0, fmax(v1, v2));

		int3 min_corner_ID = convert_int3((min_corner_AABB - min_corner_grid) / voxel_size);
		int3 max_corner_ID = convert_int3((max_corner_AABB - min_corner_grid) / voxel_size);

		//printf("thread: %d --> min_corner_AABB: (%f, %f, %f), max_corner_AABB: (%f, %f, %f), min_corner_ID: (%d, %d, %d), max_corner_ID: (%d, %d, %d)\n",
		//	i, min_corner_AABB.x, min_corner_AABB.y, min_corner_AABB.z,
		//	max_corner_AABB.x, max_corner_AABB.y, max_corner_AABB.z,
		//	min_corner_ID.x, min_corner_ID.y, min_corner_ID.z,
		//	max_corner_ID.x, max_corner_ID.y, max_corner_ID.z);

		for (int j = min_corner_ID.z; j <= max_corner_ID.z; j++)
		{
			for (int k = min_corner_ID.y; k <= max_corner_ID.y; k++)
			{
				for (int l = min_corner_ID.x; l <= max_corner_ID.x; l++)
				{
					__private int voxel_index = l + k * dim_grid.x + j * dim_grid.x * dim_grid.y;
					//printf("in thread %d ==> for (l,k,j)=(%d, %d, %d)--> center = (%f, %f, %f), voxel_index = %d\n",
					//	i, l, k, j, center.x, center.y, center.z, voxel_index);

					if (density[voxel_index] != 5)
					{
						float3 e0 = v1 - v0;
						float3 e1 = v2 - v1;
						float3 e2 = v0 - v2;
						float3 n = normalize(cross(e0, e1));

						__private float3 p = (float3)(l, k, j) * voxel_size + min_corner_grid;

						//plane test
						float3 delta_p = { voxel_size, voxel_size, voxel_size };
						float3 c = (float3)(0.0f);
						if (n.x > 0.0f)
						{
							c.x = delta_p.x;
						}
						if (n.y > 0.0f)
						{
							c.y = delta_p.y;
						}
						if (n.z > 0.0f)
						{
							c.z = delta_p.z;
						}
						float d1 = dot(n, (c - v0));
						float d2 = dot(n, ((delta_p - c) - v0));
						float n_dot_p = dot(n, p);
						if (((n_dot_p + d1) * (n_dot_p + d2)) > 0)
						{
							continue;
						};

						//2D projections
						//XY plane
						float2 n_xy_e0 = { -1.0f * e0.y, e0.x };
						float2 n_xy_e1 = { -1.0f * e1.y, e1.x };
						float2 n_xy_e2 = { -1.0f * e2.y, e2.x };
						if (n.z < 0.0f)
						{
							n_xy_e0 = -n_xy_e0;
							n_xy_e1 = -n_xy_e1;
							n_xy_e2 = -n_xy_e2;
						}
						float d_xy_e0 = -1.0f * dot(n_xy_e0, (float2)(v0.x, v0.y)) + fmax(0.0f, voxel_size * n_xy_e0.s0) + fmax(0.0f, voxel_size * n_xy_e0.s1);
						float d_xy_e1 = -1.0f * dot(n_xy_e1, (float2)(v1.x, v1.y)) + fmax(0.0f, voxel_size * n_xy_e1.s0) + fmax(0.0f, voxel_size * n_xy_e1.s1);
						float d_xy_e2 = -1.0f * dot(n_xy_e2, (float2)(v2.x, v2.y)) + fmax(0.0f, voxel_size * n_xy_e2.s0) + fmax(0.0f, voxel_size * n_xy_e2.s1);
						float2 p_xy = { p.x, p.y };
						if ((dot(n_xy_e0, p_xy) + d_xy_e0) < 0)
						{
							continue;
						}
						if ((dot(n_xy_e1, p_xy) + d_xy_e1) < 0)
						{
							continue;
						}
						if ((dot(n_xy_e2, p_xy) + d_xy_e2) < 0)
						{
							continue;
						}
						// YZ plane
						float2 n_yz_e0 = { -1.0f * e0.z, e0.y };
						float2 n_yz_e1 = { -1.0f * e1.z, e1.y };
						float2 n_yz_e2 = { -1.0f * e2.z, e2.y };
						if (n.x < 0.0f)
						{
							n_yz_e0 = -n_yz_e0;
							n_yz_e1 = -n_yz_e1;
							n_yz_e2 = -n_yz_e2;
						}
						float d_yz_e0 = -1.0f * dot(n_yz_e0, (float2)(v0.y, v0.z)) + fmax(0.0f, voxel_size * n_yz_e0.s0) + fmax(0.0f, voxel_size * n_yz_e0.s1);
						float d_yz_e1 = -1.0f * dot(n_yz_e1, (float2)(v1.y, v1.z)) + fmax(0.0f, voxel_size * n_yz_e1.s0) + fmax(0.0f, voxel_size * n_yz_e1.s1);
						float d_yz_e2 = -1.0f * dot(n_yz_e2, (float2)(v2.y, v2.z)) + fmax(0.0f, voxel_size * n_yz_e2.s0) + fmax(0.0f, voxel_size * n_yz_e2.s1);
						float2 p_yz = { p.y, p.z };
						if ((dot(n_yz_e0, p_yz) + d_yz_e0) < 0)
						{
							continue;
						}
						if ((dot(n_yz_e1, p_yz) + d_yz_e1) < 0)
						{
							continue;
						}
						if ((dot(n_yz_e2, p_yz) + d_yz_e2) < 0)
						{
							continue;
						}
						// ZX plane
						float2 n_zx_e0 = { -1.0f * e0.x, e0.z };
						float2 n_zx_e1 = { -1.0f * e1.x, e1.z };
						float2 n_zx_e2 = { -1.0f * e2.x, e2.z };
						if (n.y < 0.0f)
						{
							n_zx_e0 = -n_zx_e0;
							n_zx_e1 = -n_zx_e1;
							n_zx_e2 = -n_zx_e2;
						}
						float d_zx_e0 = -1.0f * dot(n_zx_e0, (float2)(v0.z, v0.x)) + fmax(0.0f, voxel_size * n_zx_e0.s0) + fmax(0.0f, voxel_size * n_zx_e0.s1);
						float d_zx_e1 = -1.0f * dot(n_zx_e1, (float2)(v1.z, v1.x)) + fmax(0.0f, voxel_size * n_zx_e1.s0) + fmax(0.0f, voxel_size * n_zx_e1.s1);
						float d_zx_e2 = -1.0f * dot(n_zx_e2, (float2)(v2.z, v2.x)) + fmax(0.0f, voxel_size * n_zx_e2.s0) + fmax(0.0f, voxel_size * n_zx_e2.s1);
						float2 p_zx = { p.z, p.x };
						if ((dot(n_zx_e0, p_zx) + d_zx_e0) < 0)
						{
							continue;
						}
						if ((dot(n_zx_e1, p_zx) + d_zx_e1) < 0)
						{
							continue;
						}
						if ((dot(n_zx_e2, p_zx) + d_zx_e2) < 0)
						{
							continue;
						}
						//printf("thread no. %d --> %d\n", i, voxel_index);
						density[voxel_index] = 5;
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
		while (o < dimGrid[0])
		{
			int index = o + yIndex * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1];
			if (density[index] == 5)
			{
				break;
			}
			o++;
		}

		int p = dimGrid[0] - 1;
		while (p > 0) 
		{
			int index = p + yIndex * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1];
			if (density[index] == 5)
			{
				break;
			}
			p--;
		}
		//printf("o: %d, p: %d\n", o, p);
		for (int q = o + 1; q <= p - 1; q++) {

			int index = q + yIndex * dimGrid[0] + zIndex * dimGrid[0] * dimGrid[1];
			if (density[index] != 5) 
			{
				density[index] = 2;
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