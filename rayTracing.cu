// example of ray tracing in Section 6.2.2
// Zheshu Wu, Jun 9, 2018

#define INF 2e10f

struct Sphere
{
	float r, b, g;
	float radius;
	float x, y, z;

	__device__ float hit( float ox, float oy, float *n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius)
		{
			float dz = sqrtf( radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf( radius * radius );
			return dz + z;
		}
		return -INF;
	}
};	// semicolon!

#include "cuda.h"
#include "./common/book.h"
#include "./common/cpu_bitmap.h"

#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 20
#define DIM 1024

__global__ void kernel( Sphere *s, unsigned char *ptr)
{
	// map from threadIdx/blockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = ( x - DIM / 2);
	float oy = ( y - DIM / 2);

	float r=0, g=0, b=0;
	float maxz = -INF;

	for (int i=0; i<SPHERES; i++)
	{
		float n;
		float t = s[i].hit( ox, oy, &n );
		if (t > maxz )
		{
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}


	}

	ptr[offset*4 + 0] = (int)(r * 255);
	ptr[offset*4 + 1] = (int)(g * 255);
	ptr[offset*4 + 2] = (int)(b * 255);
	ptr[offset*4 + 3] = 255;

}

int main( void )
{
	// capture the start time
	cudaEvent_t		start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

	CPUBitmap bitmap( DIM, DIM );
	unsigned char *dev_bitmap;

	// allocate memory on the GPU for the output bitmap
	cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() );
	cudaMalloc( (void**)&s, sizeof(Sphere) * SPHERES );

	// allocate temp memory, initialize it, copy to memory on the GPU
	// and then free our temp memeory
	Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES);
	for (int i=0; i<SPHERES; i++)
	{
		temp_s[i].r = rnd( 1.0f );
		temp_s[i].g = rnd( 1.0f );
		temp_s[i].b = rnd( 1.0f );
		temp_s[i].x = rnd( 1000.0f ) - 500;
		temp_s[i].y = rnd( 1000.0f ) - 500;
		temp_s[i].z = rnd( 1000.0f ) - 500;
		temp_s[i].radius = rnd( 100.0f ) + 20;
	}

	cudaMemcpy( s, temp_s, sizeof(Sphere) * SPHERES, 
				cudaMemcpyHostToDevice);
	free( temp_s );

	// generate a bitmap from our sphere data
	dim3	grids( DIM/16, DIM/16 );
	dim3	threads( 16, 16 );
	kernel<<< grids, threads >>>( dev_bitmap );

	// copy back from the GPU for display
	cudaMemcpy( bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), 
				cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();

	// free memory
	cudaFree( dev_bitmap );
	cudaFree( s );

}
