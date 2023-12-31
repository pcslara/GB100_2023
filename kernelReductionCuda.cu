#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define GCE(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess)  {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void vector_sum( double * x, double * output, int n ) {
   
   
   int gid =  blockIdx.x*blockDim.x+threadIdx.x;
   int lid =  threadIdx.x;
   int block_size = blockDim.x;
   
   extern __shared__ double partial_sum[];
   
   partial_sum[lid] = gid < n ?  x[gid] : 0;
   
   
   
   __syncthreads(); // block barrier
   
   
   for( int i = block_size/2; i > 0; i /= 2 ) {
      if(lid < i) {
         partial_sum[lid] += partial_sum[lid + i];
      }
      __syncthreads();
   }
  
   if(lid == 0) {
      output[ blockIdx.x ] = partial_sum[0];
   }
}


__global__ void vector_sum_global( double * x, double * output, int n ) {
   
   
   int gid =  blockIdx.x*blockDim.x+threadIdx.x;
   int lid =  threadIdx.x;
   int block_size = blockDim.x;
   
   
   
   __syncthreads(); // block barrier
   
   
   for( int i = block_size/2; i > 0; i /= 2 ) {
      if(lid < i) {
         x[gid] += x[gid + i];
      }
      __syncthreads();
   }
  
   if(lid == 0) {
      output[ blockIdx.x ] = x[blockIdx.x*blockDim.x];
   }
}

int main( int argc, char* argv[] )
{
 
    int n = 1024000;
    int threadsPerBlock = 256;
    int numberOfBlocks  =  (n + threadsPerBlock - 1) / threadsPerBlock;
    
    double *h_x;
    double *h_out;
    
    double *d_x;
    double *d_out;
    
    
    size_t size_x = n*sizeof(double);
    size_t size_out = numberOfBlocks*sizeof(double);
    
    
    h_x = (double*)malloc(size_x);
    h_out = (double*)malloc(size_out);
    
    
    // Allocate memory for each vector on GPU
    GCE( cudaMalloc(&d_x, size_x) );
    GCE( cudaMalloc(&d_out, size_out) );
    
    int i;
    for( i = 0; i < n; i++ ) {
        h_x[i] = 1./n; //
    }
 
    GCE( cudaMemcpy( d_x, h_x, size_x, cudaMemcpyHostToDevice));
 
  
    vector_sum<<<numberOfBlocks, threadsPerBlock,threadsPerBlock * sizeof( double ) >>>(d_x, d_out,n );
    
    // vector_sum_global<<<numberOfBlocks, threadsPerBlock,threadsPerBlock * sizeof( double ) >>>(d_x, d_out,n );
 
    GCE( cudaMemcpy( h_out, d_out, size_out, cudaMemcpyDeviceToHost ) );
 
    double sum = 0;
    
    for(i=0; i< numberOfBlocks; i++) {
        sum += h_out[i];
        //printf("partial: %lf\n", h_out[i] );    
    }
        
    printf("final result: %f\n", sum );
 
    cudaFree(d_x);
    cudaFree(d_out);
 
    free(h_x);
    free(h_out);

    return 0;
}

