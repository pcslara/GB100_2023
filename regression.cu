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

__global__ void vector_cov( double * x, double xMean, 
                            double * y, double yMean,
                            double * output, int n ) {
   int gid =  blockIdx.x*blockDim.x+threadIdx.x;
   int lid =  threadIdx.x;
   int block_size = blockDim.x;
   
   extern __shared__ double partial_sum[];
   
   partial_sum[lid] = gid < n ?  (x[gid] - xMean)*(y[gid]-yMean) : 0;
   
   
   
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


__global__ void vector_var( double * x, double xMean, 
                            double * output, int n ) {
   int gid =  blockIdx.x*blockDim.x+threadIdx.x;
   int lid =  threadIdx.x;
   int block_size = blockDim.x;
   
   extern __shared__ double partial_sum[];
   
   partial_sum[lid] = gid < n ?  (x[gid] - xMean)*(x[gid]-xMean) : 0;
   
   
   
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




int main( int argc, char* argv[] )
{
 
    int n = 102400;
    int threadsPerBlock = 256;
    int numberOfBlocks  =  (n + threadsPerBlock - 1) / threadsPerBlock;
    
    double *h_x;
    double *h_y;
    double *h_out;
    
    double *d_x;
    double *d_y;
    double *d_out;
    
    double xMean, yMean;
    double cov_xy, var_x, var_y;

    
    size_t size_xy = n*sizeof(double);
    size_t size_out = numberOfBlocks*sizeof(double);
    
    
    h_x = (double*)malloc(size_xy);
    h_y = (double*)malloc(size_xy);
    h_out = (double*)malloc(size_out);
    
    
    // Allocate memory for each vector on GPU
    GCE( cudaMalloc(&d_x, size_xy) );
    GCE( cudaMalloc(&d_y, size_xy) );
    GCE( cudaMalloc(&d_out, size_out) );
    
    int i;
    for( i = 0; i < n; i++ ) {
        h_x[i] = (double)i / n; //
        h_y[i] = 2.123 * h_x[i] + 9.8736;
    }
 
    GCE( cudaMemcpy( d_x, h_x, size_xy, cudaMemcpyHostToDevice));
    GCE( cudaMemcpy( d_y, h_y, size_xy, cudaMemcpyHostToDevice));
 
  
    vector_sum<<<numberOfBlocks, threadsPerBlock,threadsPerBlock * sizeof( double ) >>>(d_x, d_out,n );


    GCE( cudaMemcpy( h_out, d_out, size_out, cudaMemcpyDeviceToHost ) );
 
    double sum = 0;
    
    for(i=0; i< numberOfBlocks; i++) {
        sum += h_out[i];
    }
        
    xMean = sum / n;


    vector_sum<<<numberOfBlocks, threadsPerBlock,threadsPerBlock * sizeof( double ) >>>(d_y, d_out,n );


    GCE( cudaMemcpy( h_out, d_out, size_out, cudaMemcpyDeviceToHost ) );
 
    sum = 0;
    
    for(i=0; i< numberOfBlocks; i++) {
        sum += h_out[i];
    }
        
    yMean = sum / n;


    vector_cov<<<numberOfBlocks, threadsPerBlock,
     threadsPerBlock * sizeof( double ) >>>
     (d_x, xMean, d_y, yMean, d_out,n );


    GCE( cudaMemcpy( h_out, d_out, size_out, cudaMemcpyDeviceToHost ) );
 
    sum = 0;
    
    for(i=0; i< numberOfBlocks; i++) {
        sum += h_out[i];
    }
        
    cov_xy = sum;

    vector_var<<<numberOfBlocks, threadsPerBlock,
     threadsPerBlock * sizeof( double ) >>>
     (d_x, xMean, d_out,n );


    GCE( cudaMemcpy( h_out, d_out, size_out, cudaMemcpyDeviceToHost ) );
 
    sum = 0;
    
    for(i=0; i< numberOfBlocks; i++) {
        sum += h_out[i];
    }
        
    var_x = sum;


    vector_var<<<numberOfBlocks, threadsPerBlock,
     threadsPerBlock * sizeof( double ) >>>
     (d_y, yMean, d_out,n );


    GCE( cudaMemcpy( h_out, d_out, size_out, cudaMemcpyDeviceToHost ) );
 
    sum = 0;
    
    for(i=0; i< numberOfBlocks; i++) {
        sum += h_out[i];
    }
        
    var_y = sum;

    double beta  = cov_xy / var_x;
    double alpha = yMean - beta * xMean;
    double rho   = cov_xy / sqrt( var_x * var_y );

    printf("%.4lf %.4lf %.4lf\n", alpha, beta, rho );


 
    printf("xMean = %lf\n", xMean );

    cudaFree(d_x);
    cudaFree(d_out);
 
    free(h_x);
    free(h_out);

    return 0;
}

