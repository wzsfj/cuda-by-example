// linear algebra calculation on GPU devices
// By Zheshu Wu, Jun 1, 2018
#include<stdio.h>
#define N 33 * 1024

__global__ void add(int *a, int *b, int *c)
{
	// int tid = 0; // CPU zero, so we start at zero
	// while (tid < N)
	// {		
	// 	c[tid] = a[tid] + b[tid];
	// 	tid += 1; // we have one CPU, so we increment by one
	// }

	// int tid = blockIdx.x;	//handle the data at this index
	// if (tid < N)
	// {
	// 	c[tid] = a[tid] + b[tid];
	// }

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{		
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x; // increment the index
	}
}

int main(int argc, char const *argv[])
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// allocate the memory on the GPU
	cudaMalloc( (void**)&dev_a, N * sizeof(int) );
	cudaMalloc( (void**)&dev_b, N * sizeof(int) );
	cudaMalloc( (void**)&dev_c, N * sizeof(int) );

	// fill the array 'a' and 'b' on the CPU
	for (int i=0; i<N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	// add ( a, b, c );

	//copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add<<< 128, 128 >>>(dev_a, dev_b, dev_c);

	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	// display the results
	// for (int i=0; i<N; i++)
	// {
	// 	printf( "%d + %d = %d\n", a[i], b[i], c[i]);
	// }
	
	// verify that the GPU did the work we requested
	bool success = true;
	for (int i=0; i<N; i++)
	{
		if ((a[i] + b[i]) != c[i])
		{
			printf( "Error: %d + %d != %d\n", a[i], b[i], c[i] );
			success = false;
		}
	}
	if (success)	printf( "We did it!\n");


	// free the memory allocated on the GPU
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );

	return 0;
}