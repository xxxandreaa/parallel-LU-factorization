#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <ctime>
#include <chrono>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREAD 1024

using namespace std;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

// this method is using n as block width for better global memory coalescing
__global__ void compute_LU(int i, int B, int B_num, int n, double *A)
{
  __shared__ double multiplier[1];

  int x = (B * (blockIdx.x % B_num)) + threadIdx.x;
  int y = blockIdx.x / B_num + i;

  // calculate multiplier
  if (x == 0)
  {
    multiplier[0] = A[y * n + (i - 1)] / A[(i - 1) * n + (i - 1)];
    __syncthreads();
  }
  
  // update row
  A[y * n + (x + (i - 1))] = A[y * n + (x + (i - 1))] - (multiplier[0] * A[(i - 1) * n + (x + (i - 1))]); 

  // write back multiplier
  A[y * n + (i - 1)] = multiplier[0];
  
}

void print_matrix(double *A, int N, int n)
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      printf("%.5f ", A[i * n + j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    fprintf(stderr, "must provide exactly 2 arguments N output_filename\n");
    return 1;
  }
  typedef std::chrono::milliseconds ms;
  auto total_starttime = duration_cast<ms>(system_clock::now().time_since_epoch()).count();

  // parsing argument
  int N = atoi(argv[1]);
  char *out_filename = argv[2];

  // generate matrix
  // srand((unsigned)time(NULL));
  int n = N, B = N, B_num = 1;
  // if row size >= max threads in a block then do padding
  if (N >= MAX_THREAD)
  {
    B = MAX_THREAD;
    n = (B * (N / B)) + ((N % B == 0)? 0 : B);
    printf("N %d\n",N);
    printf("n %d\n",n);
    B_num = n / B;
  }
  printf("init success\n");

  double *A = (double *)malloc(n * n * sizeof(double));
  double *L = (double *)malloc(N * N * sizeof(double));

  if ((A == NULL)||(L == NULL))
  {
    printf("malloc failed\n");
    exit(1);
  }


  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      A[i * n + j] = 1 + (rand() % 10000);
      L[i * N + j] = 0;
    }
    // ensure diagonally dominant
    A[i * n + i] = A[i * n + i] + 10000 * N;
  }
  printf("alloc mem success\n");

  // do the padding
  if (N >= MAX_THREAD)
  {
    for (int i = N; i < n; ++i)
    {
      for (int j = 0; j < n; ++j)
      {
        A[i * n + j] = 1000 + i + j;
      }
      A[i * n + i] = A[i * n + i] + 10000 * n;
    }
    for (int j = N; j < n; ++j)
    {
      for (int i = 0; i < n - N; ++i)
      {
        A[i * n + j] = 1000 + i + j;
      }
    }
  }
  printf("padding success\n");

  // print matrix before lu factorization
  if (N < 11)
  {
    printf("the matrix before lu factorization is\n");
    print_matrix(A, N, n);
  }

  // allocate and copy memory to device
  double *device_A;
  cudaMalloc(&device_A, n * n * sizeof(double));
  cudaMemcpy(device_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);

  // LU factorization
  for (int i = 1; i < N; ++i)
  {
    compute_LU<<<(N - i) * B_num, B - (i - 1)>>>(i, B, B_num, n, device_A); 
  }

  // copy result back to host
  cudaMemcpy(A, device_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(device_A);

  // extract L and U
  for (int i = 1; i < N; ++i)
  {
    for (int j = i - 1; j >= 0; --j)
    {
      L[i * N + j] = A[i * n + j];
      A[i * n + j] = 0;
    }
  }

  // assign 1 to diagonal of L
  for (int i = 0; i < N; ++i)
  {
    L[i * N + i] = 1;
  }

  // print outcome
  if (N < 11)
  {
    printf("the lu factorization outcome is\n");
    printf("U is\n");
    print_matrix(A, N, n);
    printf("L is\n");
    print_matrix(L, N, N);
  }

  // write result to output file
  ofstream out_file(out_filename);
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      out_file.write((char *)&A[i * n + j], sizeof(double));
    }
  }
  for (int i = 0; i < N * N; ++i)
  {
    out_file.write((char *)&L[i], sizeof(double));
  }
  out_file.close();
  free(A);
  free(L);

  // calculate total spent time
  auto total_endtime = duration_cast<ms>(system_clock::now().time_since_epoch()).count();
  printf("total time spent for blocked lu %ld ms\n", (total_endtime - total_starttime));
}