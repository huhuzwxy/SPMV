#ifndef UTILS_CUDA_H
#define UTILS_CUDA_H

#include "common_cuda.h"

struct anonymouslib_timer {
    cudaEvent_t start_event, stop_event;

    void start() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, 0);
        cudaDeviceSynchronize();
    }

    float stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
        return elapsedTime;
    }
};

template<typename iT>
__inline__ __device__
iT binary_search_right_boundary_kernel(const iT *d_row_pointer,const iT  key_input,const iT  size){
    iT start = 0;
    iT stop  = size - 1;
    iT median;
    iT key_median;

    while (stop >= start){
        median = (stop + start) / 2;
        key_median = d_row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }
    return start;
}

template<typename T>
__inline__ __device__
void sum_32(volatile T *s_sum,const int local_id){
    s_sum[local_id] += s_sum[local_id + 16];
    s_sum[local_id] += s_sum[local_id + 8];
    s_sum[local_id] += s_sum[local_id + 4];
    s_sum[local_id] += s_sum[local_id + 2];
    s_sum[local_id] += s_sum[local_id + 1];
}

__device__ __forceinline__
double __shfl_xor1(double var, int srcLane, int width=32){
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_xor1(a.x, srcLane, width);
    a.y = __shfl_xor1(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

template<typename vT>
__forceinline__ __device__
vT sum_32_shfl(vT sum){
    #pragma unroll
    for(int mask = ANONYMOUSLIB_CSR5_OMEGA / 2 ; mask > 0 ; mask >>= 1)
        sum += __shfl_xor1(sum, mask);

    return sum;
}

// exclusive scan
template<typename T>
__inline__ __device__
void scan_32(volatile T *s_scan,const int local_id){
    int ai, bi;
    const int baseai = 2 * local_id + 1;
    const int basebi = baseai + 1;
    T temp;

    if (local_id < 16)  { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0)  { s_scan[31] = s_scan[15]; s_scan[15] = 0; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16)  { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

// exclusive scan
template<typename T>
__inline__ __device__
void scan_32_plus1(volatile T *s_scan,const int local_id){
    int ai, bi;
    const int baseai = 2 * local_id + 1;
    const int basebi = baseai + 1;
    T temp;

    if (local_id < 16)  { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0)  { s_scan[32] = s_scan[31] + s_scan[15]; s_scan[31] = s_scan[15]; s_scan[15] = 0; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16)  { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

template<typename T>
__inline__ __device__
void scan_256_plus1(volatile T *s_scan){
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    T temp;

    if (threadIdx.x < 128) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[255] += s_scan[127]; s_scan[256] = s_scan[255]; s_scan[255] = 0; temp = s_scan[127]; s_scan[127] = 0; s_scan[255] += temp; }
    if (threadIdx.x < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 128) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

template<typename iT>
__forceinline__ __device__
void fetch_x(cudaTextureObject_t d_x_tex,const iT i,double *x){
    int2 x_int2 = tex1Dfetch<int2>(d_x_tex, i);
    *x = __hiloint2double(x_int2.y, x_int2.x);
}

/*template<typename iT>
__forceinline__ __device__
void tryfetch_x(const iT i, double *x){
	cudaTextureObject_t d_x_tex;
	int2 x_int2 = tex1Dfetch<int2>(d_x_tex, i);
	*x = __hiloint2double(x_int2.y, x_int2.x);
}*/

__forceinline__ __device__
static double atomicAdd(double *addr, double val){
    double old = *addr, assumed;
    do{
        assumed = old;
        old = __longlong_as_double( atomicCAS((unsigned long long int*)addr, __double_as_longlong(assumed), __double_as_longlong(val+assumed)));
    }while(assumed != old);

    return old;
}

#endif // UTILS_CUDA_H
