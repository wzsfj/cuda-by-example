// for inner product of two vectors
// or vector dot product
// Jun 7, 2018 by Zheshu Wu

#include "./common/book.h"

#define imin(a,b) (a<b?a:b)
const int N = 33 * 1024;
const int threadsPerBlock = 256;

__global__ void dot( float *a, float *b, float *c)
{
	__shared__ float cache[threadsPerBlock];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		// increment by available numer of threads
		tid += blockDim.x * gridDim.x;
	}

	// set the cache values
	cache[cacheIndex] = temp;

	// synchronize threads in THIS BLOCK
	__syncthreads();

	// reduction of sum
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	// only need one thread to execute this assignment
	if (cacheIndex == 0)
	{
		c[blockIdx.x] = cache[0];
	}

}

const int blocksPerGrid = 
			imin( 32, (N + threadsPerBlock - 1) / threadsPerBlock);

int main( void )
{
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;

	// allocate memoery on the CPU side
	a = new float[N];
	b = new float[N];
	partial_c = new float[blocksPerGrid];

	// allocate memory on the GPU
	cudaMalloc( (void**)&dev_a, N * sizeof(float));
	cudaMalloc( (void**)&dev_b, N * sizeof(float));
	cudaMalloc( (void**)&dev_partial_c, blocksPerGrid * sizeof(float));

	// fill in the host memory with data
	for (int i=0; i<N; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	// copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy( dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	dot<<<blocksPerGrid, threadsPerBlock>>>( dev_a, dev_b, dev_partial_c);

	// copy the partial result back to CPU
	cudaMemcpy( partial_c, dev_partial_c, blocksPerGrid * sizeof(float),
				cudaMemcpyDeviceToHost);

	// finish up on the CPU side
	c = 0;
	for(int i=0; i<blocksPerGrid; i++)
	{
		c += partial_c[i];
	}

	#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	printf( " Does GPU value %.6g = %.6g?\n", c, 2 * 
				sum_squares( (float) (N-1) ));

	// free memory on the GPU side
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_partial_c );

	// compute the dot product on CPU only
	float sum_cpu = 0;
	for(int i=0; i<N; i++)
	{
		sum_cpu += a[i] * b[i];
	}
	printf("sum_cpu = %f \n", sum_cpu);

	// free the memeory on the CPU side
	delete [] a;
	delete [] b;
	delete [] partial_c;

}
