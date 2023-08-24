#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define SIZE 128*1024*1024

#define GCE(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess)  {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void hist( unsigned char *buffer, long size, unsigned int *hist ) {
    
    for( int i = 0;  i < size; i++ ) {
        hist[ buffer[i] ] ++;    
    }
}


__global__ void hist_kernel( unsigned char *buffer, long size, unsigned int *hist ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    while( i < size ) {
        atomicInc( &hist[ buffer[i] ], 1 );
        i += stride;    
    }
}


__global__ void hist_kernel_shared( unsigned char *buffer, long size, unsigned int *hist ) {
    __shared__ unsigned int sh_hist[256];
    sh_hist[threadIdx.x] = 0;
    __syncthreads();
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
        
    while( i < size ) {
        atomicAdd( &sh_hist[ buffer[i] ] , 1 );
        i += stride;    
    }
    __syncthreads();
    atomicAdd( &hist[threadIdx.x ] , sh_hist[ threadIdx.x ] );
}



int main() {
    unsigned char * h_buffer;
    unsigned int  * h_hist;
    
    
    unsigned char * d_buffer;
    unsigned int  * d_hist;
    
    h_buffer = (unsigned char *) malloc( SIZE );
    h_hist   = (unsigned int *) malloc( 256 * sizeof( unsigned int ) );
    
    
    unsigned int s = 0;
    
    for( int i = 0; i < 256; i++ ) {
        h_hist[i] = 0;
    }
    
    
    for( int i = 0; i < SIZE; i++ ) {
        h_buffer[i] = rand() & 0xFF;
    }
    
    cudaEvent_t start, stop;
    float elapsedTime;
    
    
    GCE( cudaMalloc(&d_buffer, SIZE) );
    GCE( cudaMalloc(&d_hist, 256 * sizeof( unsigned int ) ) );
    
    
    GCE( cudaMemcpy( d_buffer, h_buffer, SIZE, cudaMemcpyHostToDevice));
    GCE( cudaMemcpy( d_hist, h_hist, 256 * sizeof( unsigned int ), cudaMemcpyHostToDevice));
    //GCE( cudaMemcpy( h_hist, d_hist, 256 * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );   
    
    
    cudaEventCreate(&start);
    cudaEventRecord(start,0);
    
    hist(h_buffer, SIZE, h_hist );
 
    
    //hist( h_buffer, SIZE, h_hist );
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("[Sequencial] Elapsed time : %f ms\n" ,elapsedTime);  
    
    s = 0;
    
    for( int i = 0; i < 256; i++ ) {
        s += h_hist[i];       
        // printf("[%d] %ld\n", i, h_hist[i] );
    }
    
    if( s != SIZE ) {
        printf("Error on hist\n");
    } else {
        printf("Hist is Okay\n");
    }
    
    for( int i = 0; i < 256; i++ ) {
        h_hist[i] = 0;
    }
    GCE( cudaMemcpy( d_hist, h_hist, 256 * sizeof( unsigned int ), cudaMemcpyHostToDevice));
    
    cudaEventCreate(&start);
    cudaEventRecord(start,0);
    
    hist_kernel<<<16, 256>>>(d_buffer, SIZE, d_hist );
 
    GCE( cudaMemcpy( h_hist, d_hist, 256 * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );   
    
    //hist( h_buffer, SIZE, h_hist );
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("[Basic Kernel] Elapsed time : %f ms\n" ,elapsedTime);
    
    
    s = 0;
    for( int i = 0; i < 256; i++ ) {
        s += h_hist[i];       
        // printf("[%d] %ld\n", i, h_hist[i] );
    }
    
    if( s != SIZE ) {
        printf("Error on hist %d\n", s);
    } else {
        printf("Hist is Okay\n");
    }
 
 
    for( int i = 0; i < 256; i++ ) {
        h_hist[i] = 0;
    }
    GCE( cudaMemcpy( d_hist, h_hist, 256 * sizeof( unsigned int ), cudaMemcpyHostToDevice));
    
    
    cudaEventCreate(&start);
    cudaEventRecord(start,0);
    
    hist_kernel_shared<<<16, 256>>>(d_buffer, SIZE, d_hist );
 
    GCE( cudaMemcpy( h_hist, d_hist, 256 * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );   
    
    //hist( h_buffer, SIZE, h_hist );
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("[Kernel Shared] Elapsed time : %f ms\n" ,elapsedTime);  
    
    
    s = 0;
    for( int i = 0; i < 256; i++ ) {
        s += h_hist[i];       
        // printf("[%d] %ld\n", i, h_hist[i] );
    }
    
    if( s != SIZE ) {
        printf("Error on hist\n");
    } else {
        printf("Hist is Okay\n");
    }
    
    return 0;

}
