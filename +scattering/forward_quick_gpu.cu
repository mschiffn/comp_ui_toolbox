//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// forward_quick_gpu.cu
//
// implementation of
// the forward scattering operator (N_observations x N_points) using
// the reference spatial transfer function
//
// 0. u_M							mixed voltage signals (N_observations x N_objects)
// = forward_quick_gpu
// (
// 0. operator_born,				object of class scattering.operator_born (scalar)
// 1. gamma_kappa,                  relative spatial fluctuations in compressibility (N_points x N_objects)
// 2. index_device					device index to be used (scalar, default: 0)
// )
//
// author: Martin F. Schiffner
// date: 2019-06-29
// modified: 2019-07-08
// All rights reserved!
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// CUDA and cuBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>

// MATLAB mex interface
#include "mex.h"
#include "matrix.h"

// custom headers
#include "adjoint_quick_gpu.h"

// parallelization
#define N_THREADS_X 8
#define N_THREADS_Y 8
#define N_THREADS_PER_BLOCK 64

// version
#define REVISION "0.1"
#define DATE "2019-07-08"

// toggle debug mode
#define DEBUG_MODE 1

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// MEX gateway function
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 1.) define local variables
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// spatial discretization
	int N_points = 0;									// number of grid points
	int N_elements = 0;									// number of array elements
	int N_objects = 0;									// number of objects

	// spectral discretizations
	int N_measurements = 0;								// number of sequential pulse-echo measurements
	int N_f_unique = 0;									// number of unique frequencies
	int* N_f_unique_measurement = 0;					// number of unique frequencies in each pulse-echo measurement
	int* N_mix_measurement = NULL;						// number of mixed voltage signals in each pulse-echo measurement
	int** N_f_mix = NULL;								// number of frequencies in each mixed voltage signal
	int** N_f_mix_cs = NULL;							// number of frequencies in each mixed voltage signal (cumulative sum)
	int** N_elements_active_mix = NULL;					// number of active array elements in each mixed voltage signal
	int N_observations = 0;								// number of observations
	int* N_observations_measurement = NULL;				// number of observations in each pulse-echo measurement
	int* N_observations_measurement_cs = NULL;			// number of observations in each pulse-echo measurement (cumulative sum)

	int N_f_mix_max = 0;								// maximum number of frequencies in each mixed voltage signal

	// frequency map
	int*** indices_f_mix_to_measurement = NULL;			// indices of unique frequencies in each pulse-echo measurement for each mixed voltage signal
	int*** indices_f_mix_to_sequence = NULL;			// indices of unique frequencies for each mixed voltage signal
	double* indices_f_measurement_to_sequence_double = NULL;
	double* indices_f_mix_to_measurement_double = NULL;

} // void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
