
#include <stdio.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t GCE(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void kernel(float *a, int offset) {
  int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  float x = (float)i;
  float s = sinf(x); 
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s*s+c*c);
}

float maxError(float *a, int n) {
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv)
{
  const int blockSize = 256, nStreams = 4;
  
  const int n = 4 * 4096 * blockSize * nStreams;
  
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);
   
  float *a, *d_a;
  GCE( cudaMallocHost((void**)&a,   bytes) );      // host pinned
  GCE(     cudaMalloc((void**)&d_a, bytes) );      // device

  float ms; // elapsed time in milliseconds
  
  // create events and streams
  cudaEvent_t startEvent, stopEvent;
  cudaStream_t stream[nStreams];
  GCE( cudaEventCreate(&startEvent) );
  GCE( cudaEventCreate(&stopEvent) );
  
  for (int i = 0; i < nStreams; ++i)
    GCE( cudaStreamCreate(&stream[i]) );
  
  // baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  
  GCE( cudaEventRecord(startEvent,0) );
  GCE( cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice) );

  kernel<<<n/blockSize, blockSize>>>(d_a, 0);

  GCE( cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost) );
  GCE( cudaEventRecord(stopEvent, 0) );
  GCE( cudaEventSynchronize(stopEvent) );
  GCE( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 1: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  GCE( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    GCE( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice, 
                               stream[i]) );
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    GCE( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  GCE( cudaEventRecord(stopEvent, 0) );
  GCE( cudaEventSynchronize(stopEvent) );
  GCE( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));
  
  
  // asynchronous version 2: 
  // loop over copy, loop over kernel, loop over copy
  memset(a, 0, bytes);
  GCE( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    GCE( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
  }
  for (int i = 0; i < nStreams; ++i ){
    int offset = i * streamSize;
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  }
  
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    GCE( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  GCE( cudaEventRecord(stopEvent, 0) );
  GCE( cudaEventSynchronize(stopEvent) );
  GCE( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

 GCE( cudaEventDestroy(startEvent) );
  GCE( cudaEventDestroy(stopEvent) );
  for (int i = 0; i < nStreams; ++i)
    GCE( cudaStreamDestroy(stream[i]) );
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}
