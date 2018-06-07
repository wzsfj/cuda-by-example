# include <iostream>
// # include "book.h"

__global__ void kernel ( void )
{

}

__global__ void add( int a, int b, int *c )
{
	*c = a + b;
}

int main(int argc, char const *argv[])
{
	// kernel<<<1, 1>>>();
	// printf( "Hello, World!\n");

	// int c;
	// int *dev_c;
	// int N = 1;
	// cudaMalloc( (void**)&dev_c, N * sizeof(int) );

	// add<<<1, N>>>( 2, 7 , dev_c);

	// cudaMemcpy ( &c, dev_c, N * sizeof(int), 
	// 						cudaMemcpyDeviceToHost );
	// printf("2 + 7 = %d\n", c);
	// cudaFree( dev_c);

	/* Below are the codes for detecting CUDA capable devices and 
	*  output some useful information of the deives for references
	*  by Wu Zheshu, May 18, 2018
	*/
	int cudaDeviceCount;
	cudaGetDeviceCount( &cudaDeviceCount );
	printf("\nDetected %d CUDA device(s) on this computer\n", cudaDeviceCount);

	cudaDeviceProp prop;	// cudaDeviceProp is a built-in struct

	// list all the detected devices
	for (int i=0; i < cudaDeviceCount; i++)
	{
		cudaGetDeviceProperties( &prop, i );
		printf(" --- General Information for Device %d ---\n", i );
		printf("\tDevice name:\t\t %s\n", prop.name );
		printf("\tCompute capability:\t %d.%d\n", prop.major, prop.minor );
		printf("\tClock rate:\t\t %d\n", prop.clockRate );	
		printf(" --- Memory Information for device %d ---\n", i );
		printf("\tTotal global mem:\t %f\n", prop.totalGlobalMem );
		//[Caution] Here the type of totalGlobalMem is "size_t", might 
		// output some erroneous number if the type is not set correctly
		printf("\tTotal constant Mem:\t %ld\n", prop.totalConstMem );
		//[Caution] Here the type of totalConstMem is "size_t"
		printf(" --- MP Information for device %d ---\n", i );
		printf("\tMultiprocessor count:\t %d\n", prop.multiProcessorCount );
		printf("\tShared mem per mp:\t %ld\n", prop.sharedMemPerBlock );
		printf("\tRegisters per mp:\t %d\n", prop.regsPerBlock );
		printf("\tThreads in warp:\t %d\n", prop.warpSize );
		printf("\tMax threads per block:\t %d\n", prop.maxThreadsPerBlock );
		printf("\tMax thread dimensions:\t (%d, %d, %d)\n",
					prop.maxThreadsDim[0], prop.maxThreadsDim[1],
					prop.maxThreadsDim[2] );
		printf("\tMax grid dimensions:\t (%f, %d, %d)\n",
					prop.maxGridSize[0], prop.maxGridSize[1],
					prop.maxGridSize[2] );
		printf("\n" );
	}	

	return 0;
}