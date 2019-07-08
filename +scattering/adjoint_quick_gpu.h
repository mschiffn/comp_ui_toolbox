#include <cuda_runtime.h>

#define BYTES_PER_MEBIBYTE 1048576
#define BYTES_PER_KIBIBYTE 1024

//-------------------------------------------------------------------------
// canonical error checking in CUDA runtime API
//-------------------------------------------------------------------------
// #define checkCudaErrors( ans ) { gpuAssert( (ans), __FILE__, __LINE__ ); }
// inline void gpuAssert( cudaError_t code, const char *file, int line )
// {
// 	if  (code != cudaSuccess )
// 	{
// 		mexPrintf( "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(code), cudaGetErrorString( code ) );
// 		// reset device to clean memory before exit
// 		cudaDeviceReset();
// 		// print error message, exit program
// 		mexErrMsgIdAndTxt( "adjoint_quick_gpu:ErrorCUDA", "CUDA error!" );
// 	}
// }

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors( val ) check( (val), #val, __FILE__, __LINE__)
template <typename T>
void check( T result, char const *const func, const char *const file, int const line )
{
	if ( result )
	{
		//mexPrintf( "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		mexPrintf( "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func );
		// reset device to clean memory before exit
		cudaDeviceReset();
		// print error message, exit program
		mexErrMsgIdAndTxt( "adjoint_quick_gpu:ErrorCUDA", "CUDA error!" );
	}
}

// function definitions
void printMemInfo();

//-------------------------------------------------------------------------
// define data types
typedef float t_float_gpu;
typedef cuComplex t_float_complex_gpu;

// kernel definitions
__global__ void compute_matrix_kernel( t_float_complex_gpu* d_Phi_float_complex, int N_f_mix, int N_points, t_float_complex_gpu* d_h_ref_float_complex, size_t pitch_h_ref, int* d_indices_grid_FOV_shift_int, size_t pitch_indices_grid_FOV_shift, int index_element, int* d_indices_f_mix_to_sequence, t_float_complex_gpu* d_p_incident_measurement_float_complex, size_t pitch_p_incident_measurement, int* d_indices_f_mix_to_measurement, t_float_complex_gpu* d_prefactors_mix_float_complex, size_t pitch_prefactors_mix, int index_active );
