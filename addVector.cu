#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// nvcc -arch=compute_35 -Wno-deprecated-gpu-targets addVector.cu 
#define GCE(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#include <sys/time.h>

long long mytime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000*1000 + tv.tv_usec;
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(float *a, float *b, float *c, int n) {
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n){
        c[id] = a[id] + b[id];
    } 
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 16000000;
 
    // Host input vectors
    float *h_a;
    float *h_b;
    //Host output vector
    float *h_c;
 
    // Device input vectors
    float *d_a;
    float *d_b;
    //Device output vector
    float *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(float);
 
    // Allocate memory for each vector on host
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    GCE( cudaMalloc(&d_a, bytes) );
    GCE( cudaMalloc(&d_b, bytes) );
    GCE( cudaMalloc(&d_c, bytes) );
    
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);          
    }
 
 
    // Copy host vectors to device
    
    GCE( cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice));
    GCE( cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice));
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 512;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);
    
    // long t0 = mytime();
    // Execute the kernel
    vecAdd <<< gridSize, blockSize >>> (d_a, d_b, d_c, n);
    
    
    // Copy array back to host
    GCE( cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost ) );
    // long t1 = mytime();
    
    //printf("time = %lfms\n", ( t1 - t0)/1000. );
    
    // Sum up vector c and print result divided by n, this should equal 1 within error
    float sum = 0;
    for(i=0; i<n; i++) {
        sum += h_c[i];
    }
    printf("final result: %.18lf\n", sum/n);
    printf("%.18lf\n", (float) 1.3 );
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
