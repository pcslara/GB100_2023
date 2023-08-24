// seq cpu 5.097
#include "bitmap.h"
#include <stdio.h>
#include <math.h>

#define GCE(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class complex {
    public:
        double r;
        double i;
        
        __host__ __device__ complex( double r, double i ) : r(r), i(i) {}
        
        __host__ __device__ complex operator*( complex& x ) {
            return complex( r * x.r - i * x.i, r * x.i + i * x.r );
        }
        
        __host__ __device__ complex operator+( complex& x ) {
            return complex( r + x.r, i + x.i );
        }
        
        __host__ __device__ double abs() { return sqrt( r*r + i*i );  }
};



__device__ double juliamap( int value, double _min, double _max, int size ) {
    return _min + value * (_max - _min) / size;
}
/**
 * Zn+1 = Zn**2 + c
 */ 
__device__ int julia( double x, double y, complex c, int max_iter, double max_abs_z ) {
    
    complex z( x, y );
    int iter = 0;
    
    while( z.abs() < max_abs_z && iter < max_iter ) {
        z = z * z + c;
        iter++;
    }
    
    return iter;
}


__global__ void julia_set( double xmin, 
                  double xmax, 
                  double ymin, 
                  double ymax,
                  int width,
                  int height,
                  complex c,
                  int max_iter,
                  double max_abs_z,
                  unsigned char * buf ) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if( i < height && j < width ) {             
        double x = juliamap( i, xmin, xmax, height );
        double y = juliamap( j, ymin, ymax, width );
        
        int color = julia( x, y, c, max_iter, max_abs_z );

        buf[ 3*(i * width + j) + 0 ] = (color & 0xF)*16;            // B 
        buf[ 3*(i * width + j) + 1 ] = ((color >> 2) & 0xF)*16;     // G
        buf[ 3*(i * width + j) + 2 ] = ((color >> 3) & 0xF )*16 ;   // R
    }
                  
                  
}      

int main() {
    int width  = 2048; 
    int height = 2048;
    double xmin = -1.5; 
    double ymin = -1.5; 
    double xmax = 1.5;
    double ymax = 1.5; 
    int max_iter = 255;
    double max_abs_z = 8;
    complex c( -0.7, 0.27015 );
    
    unsigned char * buf = (unsigned char *) malloc( width * height * 3 );
    unsigned char * dev_buf;
    
    GCE( cudaMalloc( &dev_buf,  width * height * 3 ) );
    
    
    dim3 threadsPerBlock( 64, 64 );
    dim3 blocksPerGrid( (height + threadsPerBlock.x - 1) / threadsPerBlock.x, 
    (width + threadsPerBlock.y - 1)  / threadsPerBlock.y );  


    julia_set<<<threadsPerBlock, blocksPerGrid >>>( xmin, xmax, ymin, ymax, width, height, c, max_iter, max_abs_z, dev_buf );
    
    cudaMemcpy( buf, dev_buf,  width * height * 3, cudaMemcpyDeviceToHost ); 
    
       
    int ret = bmp_save((char *)"a.bmp", width, height, buf);
    
    return 0;

}
