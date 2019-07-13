//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// combined_quick_gpu.cu
//
// implementation of
// the forward scattering operator (N_observations x N_points) and
// the adjoint scattering operator (N_points x N_observations) using
// the reference spatial transfer function
//
// 0. output_matrix					mode = 1: u_M mixed voltage signals (N_observations x N_objects)
//									mode = 2: gamma_kappa adjoint relative spatial fluctuations in compressibility (N_points x N_objects)
// = combined_quick_gpu
// (
// 0. operator_born,				object of class scattering.operator_born (scalar)
// 1. mode,							mode of operation (1 = forward, 2 = adjoint)
// 2. input_matrix,					mode = 1: gamma_kappa relative spatial fluctuations in compressibility (N_points x N_objects)
//									mode = 2: u_M mixed voltage signals (N_observations x N_objects)
// )
//
// author: Martin F. Schiffner
// date: 2019-06-29
// modified: 2019-07-11
// All rights reserved!
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// The Thrust template library can do this trivially using thrust::transform, for example:
// thrust::multiplies<thrust::complex<float> > op;
// thrust::transform(thrust::device, x, x + n, y, z, op);
// would iterate over each pair of inputs from the device pointers x and y and calculate z[i] = x[i] * y[i]
// (there is probably a couple of casts you need to make to compile that, but you get the idea). But that effectively requires compilation of CUDA code within your project, and apparently you don't want that.

// TODO: mwSize vs size_t in mxMalloc
// TODO: make h_ref persistent -> persistent pointers to d_indices_grid_FOV_shift, d_prefactors_mix_float_complex
// TODO: cudaProfilerInitialize, cudaProfilerStart, cudaProfilerStop
// TODO: cudaHostAllocMapped pinned memory / cudaHostAllocPortable / cudaHostGetDevicePointer

// CUDA and cuBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>

// MATLAB mex interface
#include "mex.h"
#include "matrix.h"

// timing
#include <time.h>

// custom headers
#include "combined_quick_gpu.h"

// parallelization
#define N_THREADS_X 8
#define N_THREADS_Y 8

// version
#define REVISION "0.1"
#define DATE "2019-07-12"

// toggle debug mode
#define DEBUG_MODE 1
// #define VERBOSITY 3

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// MEX gateway function
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 1.) define local variables
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	//---------------------------------------------------------------------
	// a) host variables
	//---------------------------------------------------------------------
	// spatial discretization
	int N_points = 0;									// number of grid points
	int N_elements = 0;									// number of array elements
	int N_objects = 0;									// number of objects

	// occupied grid points
	int N_points_occupied = 0;							// number of occupied grid points
	int* indices_grid_FOV_occupied = NULL;				// indices of occupied grid points

	// grid point maps
	int** indices_grid_FOV_shift = 0;					// indices of shifted grid points for each array element

	// spectral discretizations
	int N_measurements = 0;								// number of sequential pulse-echo measurements
	int N_f_unique = 0;									// number of unique frequencies
	int* N_f_unique_measurement = 0;					// number of unique frequencies in each pulse-echo measurement
	int* N_mix_measurement = NULL;						// number of mixed voltage signals in each pulse-echo measurement
	int** N_f_mix = NULL;								// number of frequencies in each mixed voltage signal
	int** N_f_mix_cs = NULL;							// number of frequencies in each mixed voltage signal (cumulative sum)
	int** N_elements_active_mix = NULL;					// number of active array elements in each mixed voltage signal
	int*** indices_active_mix = NULL;					// indices of active array elements in each mixed voltage signal
	double* indices_active_mix_double = NULL;
	int N_observations = 0;								// number of observations
	int* N_observations_measurement = NULL;				// number of observations in each pulse-echo measurement
	int* N_observations_measurement_cs = NULL;			// number of observations in each pulse-echo measurement (cumulative sum)

	// statistics
	int N_f_unique_measurement_max = 0;					// maximum number of unique frequencies in each pulse-echo measurement
	int N_f_mix_max = 0;								// maximum number of frequencies in each mixed voltage signal

	// frequency maps
	int*** indices_f_mix_to_measurement = NULL;			// indices of unique frequencies in each pulse-echo measurement for each mixed voltage signal
	int*** indices_f_mix_to_sequence = NULL;			// indices of unique frequencies for each mixed voltage signal

	// options
	int mode = 1;										// mode of operation (1 = forward, 2 = adjoint)
	int index_device = 0;								// index of CUDA device
	int anti_aliasing = 0;								// status of the spatial anti-aliasing filter (0 = off / 1 = on)

	// dimensions of output vector
	int N_columns = 0;									// number of columns in the operator
	mwSize dimensions_output[ 2 ];

	// reference spatial transfer function
	mxArray* h_ref = NULL;
	mxComplexDouble* h_ref_complex = NULL;

	size_t size_bytes_h_ref = 0, pitch_h_ref = 0;
	t_float_complex_gpu* h_ref_float_complex = NULL;

	// indices of shifted grid points
	mxDouble* indices_grid_FOV_shift_double = NULL;

	// indices of unique frequencies
	mxDouble* indices_f_measurement_to_sequence_double = NULL;
	mxDouble* indices_f_mix_to_measurement_double = NULL;

	// incident acoustic pressure field
	mxArray* p_incident_measurement = NULL;
	mxComplexDouble * p_incident_measurement_complex = NULL;

	size_t size_bytes_p_incident_measurement = 0, pitch_p_incident_measurement = 0;
	t_float_complex_gpu *p_incident_measurement_float_complex = NULL;

	// prefactors for current mix
	mxComplexDouble * prefactors_mix_complex = NULL;

	size_t size_bytes_prefactors_mix = 0, pitch_prefactors_mix = 0;
	t_float_complex_gpu *prefactors_mix_float_complex = NULL;

	// input matrix (compressibility fluctuations / mixed voltage signals)
	mxComplexDouble* input_complex = NULL;

	// relative spatial fluctuations in compressibility
	size_t size_bytes_gamma_kappa = 0;
	t_float_complex_gpu* gamma_kappa_float_complex = NULL;

	// mixed voltage signals
	size_t size_bytes_u_M_act = 0;
	t_float_complex_gpu*** u_M_float_complex = NULL;

	// output matrix (mixed voltage signals / adjoint compressibility fluctuations)
	mxComplexDouble* output_complex = NULL;

	// misc variables
	mxDouble* size = NULL;
	mxArray* discretization = NULL;
	mxArray* discretization_spectral = NULL;
	mxArray* rx_measurement = NULL;
	mxArray* prefactors_measurement = NULL;
	mxArray* temp = NULL;

	int index_act = 0, index_src = 0, index_element = 0;

	// parallelization
	int N_blocks_x = 0, N_blocks_y = 0;					// numbers of blocks along 1st and 2nd dimension
	dim3 threadsPerBlock( N_THREADS_X, N_THREADS_Y );

	// cuBLAS settings
	const t_float_complex_gpu gemm_alpha = make_cuFloatComplex( 1.0f, 0.0f );
	const t_float_complex_gpu gemm_beta = make_cuFloatComplex( 1.0f, 0.0f );
	cublasHandle_t handle;

	//---------------------------------------------------------------------
	// b) device variables
	//---------------------------------------------------------------------
	t_float_complex_gpu* d_h_ref_float_complex = NULL;
	t_float_complex_gpu* d_p_incident_measurement_float_complex = NULL;
// TODO: convert to type t_float_complex_gpu*** for each active element in each mix of each measurement
	t_float_complex_gpu* d_prefactors_mix_float_complex = NULL;

	// input and output matrices
	t_float_complex_gpu* d_gamma_kappa_float_complex = NULL;
	t_float_complex_gpu*** d_u_M_float_complex = NULL;

	// intermediate results
	t_float_complex_gpu* d_Phi_float_complex = NULL;

	int** d_indices_grid_FOV_shift = NULL;
	int*** d_indices_f_mix_to_measurement = NULL;
	int*** d_indices_f_mix_to_sequence = NULL;

	size_t size_bytes_indices_f_act = 0;
	size_t size_bytes_Phi_max = 0;

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 2.) check arguments
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// check number of arguments and outputs
	if ( nrhs != 3 || nlhs != 1 ) mexErrMsgIdAndTxt( "combined_quick_gpu:NumberArguments", "combined_quick_gpu requires 3 inputs and 1 output." );

	//---------------------------------------------------------------------
	// a) ensure class scattering.operator_born (scalar)
	//---------------------------------------------------------------------
	// ensure class scattering.operator_born (scalar)
	if( mxIsClass( prhs[ 0 ], "scattering.operator_born" ) && mxIsScalar( prhs[ 0 ] ) )
	{

		discretization = mxGetProperty( prhs[ 0 ], 0, "discretization" );
		size = mxGetDoubles( mxGetProperty( discretization, 0, "size" ) );

		discretization_spectral = mxGetProperty( discretization, 0, "spectral" );

		// number of observations
		// N_observations = mxGetM( prhs[ 2 ] );
		N_observations = (int) size[ 0 ];
		if( DEBUG_MODE ) mexPrintf( "N_observations = %d\n", N_observations );

		// number of grid points
		// N_points = (int) mxGetScalar( mxGetProperty( mxGetProperty( mxGetProperty( discretization, 0, "spatial" ), 0, "grid_FOV" ), 0, "N_points" ) );
		N_points = (int) size[ 1 ];
		if( DEBUG_MODE ) mexPrintf( "N_points = %d\n", N_points );

		// extract indices of shifted grid points
		indices_grid_FOV_shift_double = mxGetDoubles( mxGetProperty( discretization, 0, "indices_grid_FOV_shift" ) );

		// number of array elements
		N_elements = (int) mxGetScalar( mxGetProperty( mxGetProperty( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "sequence" ), 0, "setup" ), 0, "xdc_array" ), 0, "N_elements" ) );
		if( DEBUG_MODE ) mexPrintf( "N_elements = %d\n", N_elements );

		// number of sequential pulse-echo measurements
		N_measurements = mxGetNumberOfElements( discretization_spectral );
		if( DEBUG_MODE ) mexPrintf( "N_measurements = %d\n", N_measurements );

		// allocate memory
		N_mix_measurement = (int*) mxMalloc( N_measurements * sizeof( int ) );
		N_f_unique_measurement = (int*) mxMalloc( N_measurements * sizeof( int ) );
		N_f_mix = (int**) mxMalloc( N_measurements * sizeof( int* ) );
		N_f_mix_cs = (int**) mxMalloc( N_measurements * sizeof( int* ) );
		N_elements_active_mix = (int**) mxMalloc( N_measurements * sizeof( int* ) );
		N_observations_measurement = (int*) mxMalloc( N_measurements * sizeof( int ) );
		N_observations_measurement_cs = (int*) mxCalloc( N_measurements, sizeof( int ) );
		indices_f_mix_to_measurement = (int***) mxMalloc( N_measurements * sizeof( int** ) );
		indices_f_mix_to_sequence = (int***) mxMalloc( N_measurements * sizeof( int** ) );
		indices_active_mix = (int***) mxMalloc( N_measurements * sizeof( int** ) );

		// iterate sequential pulse-echo measurements
		for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
		{

			// numbers of mixed voltage signals per measurement
			N_mix_measurement[ index_measurement ] = mxGetNumberOfElements( mxGetProperty( discretization_spectral, index_measurement, "rx" ) );
			if( DEBUG_MODE ) mexPrintf( "N_mix_measurement[%d] = %d\n", index_measurement, N_mix_measurement[ index_measurement ] );

			// allocate memory
			N_f_mix[ index_measurement ] = (int*) mxMalloc( N_mix_measurement[ index_measurement ] * sizeof( int ) );
			N_f_mix_cs[ index_measurement ] = (int*) mxCalloc( N_mix_measurement[ index_measurement ], sizeof( int ) );
			N_elements_active_mix[ index_measurement ] = (int*) mxMalloc( N_mix_measurement[ index_measurement ] * sizeof( int ) );
			indices_f_mix_to_measurement[ index_measurement ] = (int**) mxMalloc( N_mix_measurement[ index_measurement ] * sizeof( int* ) );
			indices_f_mix_to_sequence[ index_measurement ] = (int**) mxMalloc( N_mix_measurement[ index_measurement ] * sizeof( int* ) );
			indices_active_mix[ index_measurement ] = (int**) mxMalloc( N_mix_measurement[ index_measurement ] * sizeof( int* ) );

			// map unique frequencies of pulse-echo measurement to global unique frequencies
			indices_f_measurement_to_sequence_double = mxGetDoubles( mxGetCell( mxGetProperty( discretization, 0, "indices_f_to_unique" ), (mwIndex) index_measurement ) );
			
			// number of unique frequencies in current measurement
			N_f_unique_measurement[ index_measurement ] = mxGetM( mxGetCell( mxGetProperty( discretization, 0, "indices_f_to_unique" ), (mwIndex) index_measurement ) );
			if( DEBUG_MODE ) mexPrintf( "N_f_unique_measurement[%d] = %d\n", index_measurement, N_f_unique_measurement[ index_measurement ] );

			N_observations_measurement[ index_measurement ] = 0;

			// iterate mixed voltage signals
			for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
			{

				// number of frequencies in current mix
				N_f_mix[ index_measurement ][ index_mix ] = mxGetM( mxGetProperty( mxGetProperty( mxGetCell( mxGetProperty( discretization, 0, "prefactors" ), (mwIndex) index_measurement ), index_mix, "samples" ), 0, "values" ) );
				if( DEBUG_MODE ) mexPrintf( "N_f_mix[%d][%d] = %d\n", index_measurement, index_mix, N_f_mix[ index_measurement ][ index_mix ] );

				// maximum number of frequencies in each mixed voltage signal
				if( N_f_mix[ index_measurement ][ index_mix ] > N_f_mix_max ) N_f_mix_max = N_f_mix[ index_measurement ][ index_mix ];

				// number of frequencies in each mixed voltage signal (cumulative sum)
				if( index_mix > 0 ) N_f_mix_cs[ index_measurement ][ index_mix ] = N_f_mix_cs[ index_measurement ][ index_mix - 1 ] + N_f_mix[ index_measurement ][ index_mix - 1 ];
				if( DEBUG_MODE ) mexPrintf( "N_f_mix_cs[%d][%d] = %d\n", index_measurement, index_mix, N_f_mix_cs[ index_measurement ][ index_mix ] );

				// number of active array elements in each mixed voltage signal
				rx_measurement = mxGetProperty( mxGetProperty( discretization_spectral, index_measurement, "rx" ), index_mix, "indices_active" );
				N_elements_active_mix[ index_measurement ][ index_mix ] = mxGetNumberOfElements( rx_measurement );
				if( DEBUG_MODE ) mexPrintf( "N_elements_active_mix[%d][%d] = %d\n", index_measurement, index_mix, N_elements_active_mix[ index_measurement ][ index_mix ] );

				// indices of active array elements for each mixed voltage signal
				indices_active_mix_double = mxGetDoubles( rx_measurement );

				// map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
				indices_f_mix_to_measurement_double = mxGetDoubles( mxGetCell( mxGetProperty( discretization_spectral, index_measurement, "indices_f_to_unique" ), (mwIndex) index_mix ) );

				// allocate memory
				indices_f_mix_to_measurement[ index_measurement ][ index_mix ] = (int*) mxMalloc( N_f_mix[ index_measurement ][ index_mix ] * sizeof( int ) );
				indices_f_mix_to_sequence[ index_measurement ][ index_mix ] = (int*) mxMalloc( N_f_mix[ index_measurement ][ index_mix ] * sizeof( int ) );
				indices_active_mix[ index_measurement ][ index_mix ] = (int*) mxMalloc( N_elements_active_mix[ index_measurement ][ index_mix ] * sizeof( int ) );

				// iterate frequencies
				for( int index_f = 0; index_f < N_f_mix[ index_measurement ][ index_mix ]; index_f++ )
				{

					// indices of unique frequencies in each pulse-echo measurement for each mixed voltage signal
					indices_f_mix_to_measurement[ index_measurement ][ index_mix ][ index_f ] = (int) indices_f_mix_to_measurement_double[ index_f ] - 1;
					if( DEBUG_MODE ) mexPrintf( "indices_f_mix_to_measurement[%d][%d][%d] = %d\n", index_measurement, index_mix, index_f, indices_f_mix_to_measurement[ index_measurement ][ index_mix ][ index_f ] );

					// indices of unique frequencies for each mixed voltage signal
					indices_f_mix_to_sequence[ index_measurement ][ index_mix ][ index_f ] = (int) indices_f_measurement_to_sequence_double[ indices_f_mix_to_measurement[ index_measurement ][ index_mix ][ index_f ] ] - 1;
					if( DEBUG_MODE ) mexPrintf( "indices_f_mix_to_sequence[%d][%d][%d] = %d\n", index_measurement, index_mix, index_f, indices_f_mix_to_sequence[ index_measurement ][ index_mix ][ index_f ] );

				} // for( int index_f = 0; index_f < N_f_mix[ index_measurement ][ index_mix ]; index_f++ )

				// iterate active elements
				for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement ][ index_mix ]; index_active++ )
				{

					// indices of active array elements for each mixed voltage signal
					indices_active_mix[ index_measurement ][ index_mix ][ index_active ] = (int) indices_active_mix_double[ index_active ] - 1;
					if( DEBUG_MODE ) mexPrintf( "indices_active_mix[%d][%d][%d] = %d\n", index_measurement, index_mix, index_active, indices_active_mix[ index_measurement ][ index_mix ][ index_active ] );

				} // for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement ][ index_mix ]; index_active++ )

				// number of observations in each pulse-echo measurement
				N_observations_measurement[ index_measurement ] += N_f_mix[ index_measurement ][ index_mix ];

			} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

			if( index_measurement > 0 ) N_observations_measurement_cs[ index_measurement ] = N_observations_measurement_cs[ index_measurement - 1 ] + N_observations_measurement[ index_measurement - 1 ];

			if( DEBUG_MODE )
			{
				mexPrintf( "N_observations_measurement[%d] = %d\n", index_measurement, N_observations_measurement[ index_measurement ] );
				mexPrintf( "N_observations_measurement_cs[%d] = %d\n", index_measurement, N_observations_measurement_cs[ index_measurement ] );
				mexPrintf( "N_f_mix_max = %d\n", N_f_mix_max );
			}

		} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )

		// index of CUDA device
		index_device = (int) mxGetScalar( mxGetProperty( mxGetProperty( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "options" ), 0, "momentary" ), 0, "gpu" ), 0, "index" ) );

		// spatial aliasing (workaround to read enumerated value)
		mxArray* mxi = NULL;
		temp = mxGetProperty( mxGetProperty( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "options" ), 0, "momentary" ), 0, "anti_aliasing" ), 0, "status" );
		mexCallMATLAB( 1, &mxi, 1, &temp, "char" );
		if( !strcmp( mxArrayToString( mxi ), "on" ) ) anti_aliasing = 1;
		if( DEBUG_MODE ) mexPrintf( "anti_aliasing = %d\n", anti_aliasing );

	}
	else
	{
		mexErrMsgIdAndTxt( "combined_quick_gpu:NoOperatorBorn", "operator_born must be a single scattering.operator_born!" );
	} // if( mxIsClass( prhs[ 0 ], "scattering.operator_born" ) && mxIsScalar( prhs[ 0 ] ) )

	//---------------------------------------------------------------------
	// b) mode of operation
	//---------------------------------------------------------------------
	if( mxIsNumeric( prhs[ 1 ] ) && mxIsScalar( prhs[ 1 ] ) && !mxIsComplex( prhs[ 1 ] ) )
	{

		mode = (int) mxGetScalar( prhs[ 1 ] );
		if( DEBUG_MODE ) mexPrintf( "mode = %d\n", mode );
		N_columns = ( mode == 1 ) ? N_points : N_observations;

	}
	else{
		mexErrMsgIdAndTxt( "combined_quick_gpu:NoNumericScalarMode", "mode must be a numeric scalar!" );
	}

	//---------------------------------------------------------------------
	// c) input matrix (N_columns x N_objects)
	//---------------------------------------------------------------------
	// ensure numeric matrix w/ correct number of rows
	if( mxIsNumeric( prhs[ 2 ] ) && mxGetNumberOfDimensions( prhs[ 2 ] ) == 2 && mxGetM( prhs[ 2 ] ) == N_columns )
	{

		// check for imaginary part
		if( mxIsComplex( prhs[ 2 ] ) )
		{
			// extract input matrix
			input_complex = mxGetComplexDoubles( prhs[ 2 ] );
		}
		else
		{
// TODO: add zero imaginary part
			mexErrMsgIdAndTxt( "combined_quick_gpu:NoNumericMatrix", "u_M must be a numeric matrix!" );
		}

		// number of objects
		N_objects = mxGetN( prhs[ 2 ] );
		if( DEBUG_MODE ) mexPrintf( "N_objects = %d\n", N_objects );

	}
	else
	{

		mexErrMsgIdAndTxt( "combined_quick_gpu:NoNumericMatrix", "input must be a numeric matrix with xxx rows!" );

	} // if( mxIsNumeric( prhs[ 2 ] ) && mxGetNumberOfDimensions( prhs[ 2 ] ) == 2 && mxGetM( prhs[ 2 ] ) == N_observations )

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 3.) MATLAB output
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// allocate workspace memory
	dimensions_output[ 0 ] = ( mode == 1 ) ? N_observations : N_points;
	dimensions_output[ 1 ] = N_objects;

	plhs[ 0 ] = mxCreateNumericArray( 2, dimensions_output, mxDOUBLE_CLASS, mxCOMPLEX );
	output_complex = mxGetComplexDoubles( plhs[ 0 ] );

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 4.) detect occupied grid points
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// allocate memory
	indices_grid_FOV_occupied = (int*) mxMalloc( N_points * sizeof( int ) );

	// set number of occupied grid points
	N_points_occupied = ( mode == 1 ) ? 0 : N_points;

	// iterate grid points
	if( DEBUG_MODE ) mexPrintf( "detecting occupied grid points...\n" );
	for( int index_point = 0; index_point < N_points; index_point++ )
	{

		if( mode == 1)
		{
			// iterate objects
			for( int index_object = 0; index_object < N_objects; index_object++ )
			{

				index_act = index_object * N_points + index_point;
			
				// check occupancy
				if( ( input_complex[ index_act ].real > EPS_POS ) || ( input_complex[ index_act ].real < EPS_NEG ) || ( input_complex[ index_act ].imag > EPS_POS ) || ( input_complex[ index_act ].imag < EPS_NEG ) )
				{
					indices_grid_FOV_occupied[ N_points_occupied ] = index_point;
					if( DEBUG_MODE ) mexPrintf( "\tindices_grid_FOV_occupied[%d] = %d\n", N_points_occupied, index_point );
					N_points_occupied++;
					break;
				}

			} // for( int index_object = 0; index_object < N_objects; index_object++ )
		}
		else
		{
			indices_grid_FOV_occupied[ index_point ] = index_point;
		} // if( mode == 1)

	} // for( int index_point = 0; index_point < N_points; index_point++ )
	if( DEBUG_MODE ) mexPrintf( "done!\n" );
	if( DEBUG_MODE ) mexPrintf( "N_points_occupied = %d\n", N_points_occupied );

	// quick exit for N_points_occupied == 0
	if( N_points_occupied == 0 ) return;

	// reallocate memory for indices of occupied grid points
	if( N_points_occupied < N_points ) indices_grid_FOV_occupied = (int*) mxRealloc( indices_grid_FOV_occupied, N_points_occupied * sizeof( int ) );

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 4.) check number of GPUs and their capabilities, print GPU information
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// get CUDA info
	// cuInfo cuda_info;
	// get_device_info( index_device, &cuda_info );
	// if( DEBUG_MODE ) print_device_info( index_device, cuda_info );

	// set device to operate on
	checkCudaErrors( cudaSetDevice( index_device ) );

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 5.) convert to float
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	//------------------------------------------------------------------------------------------------------------------------------------------
	// a) extract reference spatial transfer function
	//------------------------------------------------------------------------------------------------------------------------------------------
	// extract reference spatial transfer function
	if( anti_aliasing )
	{
		// activate spatial anti-aliasing filter
		h_ref = mxGetProperty( mxGetProperty( mxGetProperty( discretization, 0, "h_ref_aa" ), 0, "samples" ), 0, "values" );
		if( DEBUG_MODE ) mexPrintf( "selected spatial transfer function h_ref_aa\n" );
	}
	else
	{
		// deactivate spatial anti-aliasing filter
		h_ref = mxGetProperty( mxGetProperty( mxGetProperty( discretization, 0, "h_ref" ), 0, "samples" ), 0, "values" );
		if( DEBUG_MODE ) mexPrintf( "selected spatial transfer function h_ref\n" );
	}
	h_ref_complex = mxGetComplexDoubles( h_ref );

	// number of unique frequencies
	N_f_unique = mxGetM( h_ref );
	if( DEBUG_MODE ) mexPrintf( "N_f_unique = %d\n", N_f_unique );

	//------------------------------------------------------------------------------------------------------------------------------------------
	// b) convert h_ref_complex to float (N_f_unique x N_points)
	//------------------------------------------------------------------------------------------------------------------------------------------
	// compute size
	size_bytes_h_ref = N_f_unique * N_points * sizeof( t_float_complex_gpu );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_h_ref = %.2f MiB (%zu B)\n", ( ( double ) size_bytes_h_ref ) / BYTES_PER_MEBIBYTE, size_bytes_h_ref );

	// allocate memory
	h_ref_float_complex = (t_float_complex_gpu *) mxMalloc( size_bytes_h_ref );

	// iterate elements
	if( DEBUG_MODE ) mexPrintf( "converting h_ref_complex to float..." );
	for ( int index_point = 0; index_point < N_points; index_point++ )
	{
		for ( int index_f = 0; index_f < N_f_unique; index_f++ )
		{
			index_act = index_point * N_f_unique + index_f;
			h_ref_float_complex[ index_act ].x = (t_float_gpu) h_ref_complex[ index_act ].real;
			h_ref_float_complex[ index_act ].y = (t_float_gpu) h_ref_complex[ index_act ].imag;
		}
	}
	if( DEBUG_MODE ) mexPrintf( "done!\n" );

	//------------------------------------------------------------------------------------------------------------------------------------------
	// c) convert indices_grid_FOV_shift_double to int (N_points_occupied x N_elements)
	//------------------------------------------------------------------------------------------------------------------------------------------
	// allocate memory
	indices_grid_FOV_shift = (int**) mxMalloc( N_elements * sizeof( int* ) );

	// iterate array elements
	if( DEBUG_MODE ) mexPrintf( "converting indices_grid_FOV_shift_double to int..." );
	for( int index_element = 0; index_element < N_elements; index_element++ )
	{

		// allocate memory
		indices_grid_FOV_shift[ index_element ] = (int*) mxMalloc( N_points_occupied * sizeof( int ) );

		// iterate occupied grid points
		for( int index_point_occupied = 0; index_point_occupied < N_points_occupied; index_point_occupied++ )
		{
			index_src = index_element * N_points + indices_grid_FOV_occupied[ index_point_occupied ];
			indices_grid_FOV_shift[ index_element ][ index_point_occupied ] = (int) indices_grid_FOV_shift_double[ index_src ] - 1;
		}

	} // for( int index_element = 0; index_element < N_elements; index_element++ )
	if( DEBUG_MODE ) mexPrintf( "done!\n" );

	//------------------------------------------------------------------------------------------------------------------------------------------
	// d)
	//------------------------------------------------------------------------------------------------------------------------------------------

	//------------------------------------------------------------------------------------------------------------------------------------------
	// e) convert input_complex to float
	//------------------------------------------------------------------------------------------------------------------------------------------
	// compute size
	size_bytes_gamma_kappa = N_points_occupied * N_objects * sizeof( t_float_complex_gpu );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_gamma_kappa = %.2f MiB (%zu B)\n", ( ( double ) size_bytes_gamma_kappa ) / BYTES_PER_MEBIBYTE, size_bytes_gamma_kappa );

	if( mode == 1 )
	{

		//--------------------------------------------------------------------------------------------------------------------------------------
		// i.) forward scattering (input_complex is N_points x N_objects)
		//--------------------------------------------------------------------------------------------------------------------------------------
		// allocate memory
		gamma_kappa_float_complex = (t_float_complex_gpu*) mxMalloc( size_bytes_gamma_kappa );

		// iterate elements
		if( DEBUG_MODE ) mexPrintf( "converting input_complex to float..." );
		for( int index_object = 0; index_object < N_objects; index_object++ )
		{
			for( int index_point_occupied = 0; index_point_occupied < N_points_occupied; index_point_occupied++ )
			{
				index_act = index_object * N_points_occupied + index_point_occupied;
				index_src = index_object * N_points + indices_grid_FOV_occupied[ index_point_occupied ];
				gamma_kappa_float_complex[ index_act ].x = (t_float_gpu) input_complex[ index_src ].real;
				gamma_kappa_float_complex[ index_act ].y = (t_float_gpu) input_complex[ index_src ].imag;
			}
		}
		if( DEBUG_MODE ) mexPrintf( "done!\n" );

	}
	else
	{

		//--------------------------------------------------------------------------------------------------------------------------------------
		// ii.) adjoint scattering (input_complex is N_observations x N_objects)
		//--------------------------------------------------------------------------------------------------------------------------------------
		// allocate memory
		u_M_float_complex = (t_float_complex_gpu***) mxMalloc( N_measurements * sizeof( t_float_complex_gpu** ) );

		// iterate sequential pulse-echo measurements
		if( DEBUG_MODE ) mexPrintf( "converting input_complex to float..." );
		for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
		{

			// allocate memory
			u_M_float_complex[ index_measurement ] = (t_float_complex_gpu**) mxMalloc( N_mix_measurement[ index_measurement ] * sizeof( t_float_complex_gpu* ) );

			// iterate mixed voltage signals
			for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
			{

				// allocate memory
				u_M_float_complex[ index_measurement ][ index_mix ] = (t_float_complex_gpu*) mxMalloc( N_f_mix[ index_measurement ][ index_mix ] * N_objects * sizeof( t_float_complex_gpu ) );

				// iterate objects
				for( int index_object = 0; index_object < N_objects; index_object++ )
				{

					// iterate frequencies
					for( int index_f = 0; index_f < N_f_mix[ index_measurement ][ index_mix ]; index_f++ )
					{

						// compute destination index
						index_act = index_object * N_f_mix[ index_measurement ][ index_mix ] + index_f;

						// compute source index
						index_src = N_observations_measurement_cs[ index_measurement ] + N_f_mix_cs[ index_measurement ][ index_mix ] + index_object * N_observations + index_f;

						u_M_float_complex[ index_measurement ][ index_mix ][ index_act ].x = (t_float_gpu) input_complex[ index_src ].real;
						u_M_float_complex[ index_measurement ][ index_mix ][ index_act ].y = (t_float_gpu) input_complex[ index_src ].imag;

					} // for( int index_f = 0; index_f < N_f_mix[ index_measurement ][ index_mix ]; index_f++ )

				} // for( int index_object = 0; index_object < N_objects; index_object++ )

			} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

		} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
		if( DEBUG_MODE ) mexPrintf( "done!\n" );

	} // if( mode == 1 )

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 6.) allocate and initialize device memory
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	//---------------------------------------------------------------------
	// a) reference spatial transfer function (N_f_unique x N_points)
	//---------------------------------------------------------------------
	// allocate device memory
	checkCudaErrors( cudaMallocPitch( (void **) &d_h_ref_float_complex, &pitch_h_ref, N_f_unique * sizeof( t_float_complex_gpu ), N_points ) );

	//---------------------------------------------------------------------
	// b) indices (N_points_occupied x N_elements)
	//---------------------------------------------------------------------
	d_indices_grid_FOV_shift = (int**) mxMalloc( N_elements * sizeof( int* ) );

	// iterate array elements
	for( int index_element = 0; index_element < N_elements; index_element++ )
	{

		// allocate device memory
		checkCudaErrors( cudaMalloc( (void**) &( d_indices_grid_FOV_shift[ index_element ] ), N_points_occupied * sizeof( int ) ) );

	} // for( int index_element = 0; index_element < N_elements; index_element++ )

	//---------------------------------------------------------------------
	// c) allocate memory for d_p_incident_measurement_float_complex (use maximum size: N_f_unique_measurement_max x N_points_occupied)
	//---------------------------------------------------------------------
	// compute size
	// allocate memory
	// device memory status

	//---------------------------------------------------------------------
	// d) d_Phi_float_complex (use maximum size: N_f_mix_max x N_points_occupied)
	//---------------------------------------------------------------------
	// compute size
	size_bytes_Phi_max = N_f_mix_max * N_points_occupied * sizeof( t_float_complex_gpu );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_Phi_max = %.2f MiB (%zu B)\n", ( (double) size_bytes_Phi_max ) / BYTES_PER_MEBIBYTE, size_bytes_Phi_max );

	// allocate memory
	checkCudaErrors( cudaMalloc( (void **) &d_Phi_float_complex, size_bytes_Phi_max ) );

	// device memory status
	if( DEBUG_MODE ) printMemInfo();

	//---------------------------------------------------------------------
	// e) relative spatial fluctuations in compressibility
	//---------------------------------------------------------------------
	checkCudaErrors( cudaMalloc( (void **) &d_gamma_kappa_float_complex, size_bytes_gamma_kappa ) );
	checkCudaErrors( cudaMemset( d_gamma_kappa_float_complex, 0, size_bytes_gamma_kappa ) );

	//---------------------------------------------------------------------
	// f) mixed voltage signals
	//---------------------------------------------------------------------
	d_u_M_float_complex = (t_float_complex_gpu***) mxMalloc( N_measurements * sizeof( t_float_complex_gpu** ) );

	// iterate sequential pulse-echo measurements
	for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
	{

		// allocate memory
		d_u_M_float_complex[ index_measurement ] = (t_float_complex_gpu**) mxMalloc( N_mix_measurement[ index_measurement ] * sizeof( t_float_complex_gpu* ) );

		// iterate mixed voltage signals
		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
		{

			// compute size
			size_bytes_u_M_act = N_f_mix[ index_measurement ][ index_mix ] * N_objects * sizeof( t_float_complex_gpu );

			// allocate and initialize device memory
			checkCudaErrors( cudaMalloc( (void **) &( d_u_M_float_complex[ index_measurement ][ index_mix ] ), size_bytes_u_M_act ) );
			checkCudaErrors( cudaMemset( d_u_M_float_complex[ index_measurement ][ index_mix ], 0, size_bytes_u_M_act ) );

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

	} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 7.) copy data to the device
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	//---------------------------------------------------------------------
	// a) reference spatial transfer function (N_f_unique x N_points)
	//---------------------------------------------------------------------
	checkCudaErrors( cudaMemcpy2D( d_h_ref_float_complex, pitch_h_ref, h_ref_float_complex, N_f_unique * sizeof( t_float_complex_gpu ), N_f_unique * sizeof( t_float_complex_gpu ), N_points, cudaMemcpyHostToDevice ) );

	// clean-up host memory
	mxFree( h_ref_float_complex );

	// register an exit function
	// mexAtExit( cleanup );

	// device memory status
	if( DEBUG_MODE ) printMemInfo();

	//---------------------------------------------------------------------
	// b) copy indices_grid_FOV_shift to the device
	//---------------------------------------------------------------------
	// iterate array elements
	for( int index_element = 0; index_element < N_elements; index_element++ )
	{
		checkCudaErrors( cudaMemcpy( d_indices_grid_FOV_shift[ index_element ], indices_grid_FOV_shift[ index_element ], N_points_occupied * sizeof( int ), cudaMemcpyHostToDevice ) );
	} // for( int index_element = 0; index_element < N_elements; index_element++ )

	// device memory status
	if( DEBUG_MODE ) printMemInfo();

	//---------------------------------------------------------------------
	// c) copy indices_f_mix_to_measurement and indices_f_mix_to_sequence to the device
	//---------------------------------------------------------------------
	// allocate memory
	d_indices_f_mix_to_measurement = (int***) mxMalloc( N_measurements * sizeof( int** ) );
	d_indices_f_mix_to_sequence = (int***) mxMalloc( N_measurements * sizeof( int** ) );

	// iterate sequential pulse-echo measurements
	for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
	{

		// allocate memory
		d_indices_f_mix_to_measurement[ index_measurement ] = (int**) mxMalloc( N_mix_measurement[ index_measurement ] * sizeof( t_float_complex_gpu* ) );
		d_indices_f_mix_to_sequence[ index_measurement ] = (int**) mxMalloc( N_mix_measurement[ index_measurement ] * sizeof( t_float_complex_gpu* ) );

		// iterate mixed voltage signals
		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
		{

			// compute size
			size_bytes_indices_f_act = N_f_mix[ index_measurement ][ index_mix ] * sizeof( int );

			// allocate device memory
			checkCudaErrors( cudaMalloc( (void **) &( d_indices_f_mix_to_measurement[ index_measurement ][ index_mix ] ), size_bytes_indices_f_act ) );
			checkCudaErrors( cudaMalloc( (void **) &( d_indices_f_mix_to_sequence[ index_measurement ][ index_mix ] ), size_bytes_indices_f_act ) );

			// copy data
			checkCudaErrors( cudaMemcpy( d_indices_f_mix_to_measurement[ index_measurement ][ index_mix ], indices_f_mix_to_measurement[ index_measurement ][ index_mix ], size_bytes_indices_f_act, cudaMemcpyHostToDevice ) );
			checkCudaErrors( cudaMemcpy( d_indices_f_mix_to_sequence[ index_measurement ][ index_mix ], indices_f_mix_to_sequence[ index_measurement ][ index_mix ], size_bytes_indices_f_act, cudaMemcpyHostToDevice ) );

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

	} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )

	// device memory status
	if( DEBUG_MODE ) printMemInfo();

	//---------------------------------------------------------------------
	// d) copy input matrix to the device
	//---------------------------------------------------------------------
	if( mode == 1 )
	{

		//---------------------------------------------------------------------
		// i.) copy gamma_kappa_float_complex to the device
		//---------------------------------------------------------------------
		checkCudaErrors( cudaMemcpy( d_gamma_kappa_float_complex, gamma_kappa_float_complex, size_bytes_gamma_kappa, cudaMemcpyHostToDevice ) );

	}
	else
	{

		//---------------------------------------------------------------------
		// ii) copy u_M_float_complex to the device
		//---------------------------------------------------------------------
		// iterate sequential pulse-echo measurements
		for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
		{

			// iterate mixed voltage signals
			for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
			{

				// compute size
				size_bytes_u_M_act = N_f_mix[ index_measurement ][ index_mix ] * N_objects * sizeof( t_float_complex_gpu );

				// copy data
				checkCudaErrors( cudaMemcpy( d_u_M_float_complex[ index_measurement ][ index_mix ], u_M_float_complex[ index_measurement ][ index_mix ], size_bytes_u_M_act, cudaMemcpyHostToDevice ) );

			} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

		} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )

	} // if( mode == 1 )

	// device memory status
	if( DEBUG_MODE ) printMemInfo();

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 8.) compute adjoint fluctuations
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// print status
	// time_start = tic;
	// str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
	// mexPrintf( "\t %s: quick adjoint scattering (GPU, Born approximation, single precision, kappa)...", str_date_time );

	// create cuBLAS handle
	checkCudaErrors( cublasCreate( &handle ) );

	// iterate sequential pulse-echo measurements
	for ( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
	{

		//-----------------------------------------------------------------
		// a) extract incident acoustic pressure field and numbers of frequencies
		//-----------------------------------------------------------------
		// extract incident acoustic pressure field
		p_incident_measurement = mxGetProperty( mxGetProperty( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "incident_waves" ), index_measurement, "p_incident" ), 0, "samples" ), 0, "values" );
		p_incident_measurement_complex = mxGetComplexDoubles( p_incident_measurement );

		//-----------------------------------------------------------------
		// b) convert p_incident_measurement to float (N_f_unique_measurement[ index_measurement ] x N_points_occupied)
		//-----------------------------------------------------------------
		// compute size
		size_bytes_p_incident_measurement = N_f_unique_measurement[ index_measurement ] * N_points_occupied * sizeof( t_float_complex_gpu );
		if( DEBUG_MODE ) mexPrintf( "size_bytes_p_incident_measurement = %.2f MiB (%zu B)\n", ( (double) size_bytes_p_incident_measurement ) / BYTES_PER_MEBIBYTE, size_bytes_p_incident_measurement );

		// allocate memory
		p_incident_measurement_float_complex = (t_float_complex_gpu *) mxMalloc( size_bytes_p_incident_measurement );

		// iterate elements
		if( DEBUG_MODE ) mexPrintf( "converting p_incident_measurement to float..." );
		for ( int index_point_occupied = 0; index_point_occupied < N_points_occupied; index_point_occupied++ )
		{
			for ( int index_f = 0; index_f < N_f_unique_measurement[ index_measurement ]; index_f++ )
			{
				index_act = index_point_occupied * N_f_unique_measurement[ index_measurement ] + index_f;
				index_src = indices_grid_FOV_occupied[ index_point_occupied ] * N_f_unique_measurement[ index_measurement ] + index_f;
				p_incident_measurement_float_complex[ index_act ].x = (t_float_gpu) p_incident_measurement_complex[ index_src ].real;
				p_incident_measurement_float_complex[ index_act ].y = (t_float_gpu) p_incident_measurement_complex[ index_src ].imag;
			}
		}
		if( DEBUG_MODE ) mexPrintf( "done!\n" );

		//-----------------------------------------------------------------
		// c) copy p_incident_measurement_float_complex to the device
		//-----------------------------------------------------------------
		// allocate memory
		checkCudaErrors( cudaMallocPitch( (void **) &d_p_incident_measurement_float_complex, &pitch_p_incident_measurement, N_f_unique_measurement[ index_measurement ] * sizeof( t_float_complex_gpu ), N_points_occupied ) );

		// copy data
		checkCudaErrors( cudaMemcpy2D( d_p_incident_measurement_float_complex, pitch_p_incident_measurement, p_incident_measurement_float_complex, N_f_unique_measurement[ index_measurement ] * sizeof( t_float_complex_gpu ), N_f_unique_measurement[ index_measurement ] * sizeof( t_float_complex_gpu ), N_points_occupied, cudaMemcpyHostToDevice ) );

		// clean-up host memory
		mxFree( p_incident_measurement_float_complex );

		// device memory status
		if( DEBUG_MODE ) printMemInfo();

		//-----------------------------------------------------------------
		// d)
		//-----------------------------------------------------------------
		// extract prefactors for all mixes
		prefactors_measurement = mxGetCell( mxGetProperty( discretization, 0, "prefactors" ), (mwIndex) index_measurement );

		//-----------------------------------------------------------------
		// e)
		//-----------------------------------------------------------------
		// iterate mixed voltage signals
		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
		{

			//-------------------------------------------------------------
			// i.) extract prefactors for current mix
			//-------------------------------------------------------------
			// extract prefactors_mix_complex
			prefactors_mix_complex = mxGetComplexDoubles( mxGetProperty( mxGetProperty( prefactors_measurement, index_mix, "samples" ), 0, "values" ) );

			//-------------------------------------------------------------
			// ii.) convert prefactors_mix_complex to float
			//-------------------------------------------------------------
			// compute size
			size_bytes_prefactors_mix = N_f_mix[ index_measurement ][ index_mix ] * N_elements_active_mix[ index_measurement ][ index_mix ] * sizeof( t_float_complex_gpu );
			if( DEBUG_MODE ) mexPrintf( "size_bytes_prefactors_mix = %.2f kiB (%zu B)\n", ( (double) size_bytes_prefactors_mix ) / BYTES_PER_KIBIBYTE, size_bytes_prefactors_mix );

			// allocate memory
			prefactors_mix_float_complex = (t_float_complex_gpu *) mxMalloc( size_bytes_prefactors_mix );

			// iterate elements
			if( DEBUG_MODE ) mexPrintf( "converting prefactors_mix_complex to float..." );
			for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement ][ index_mix ]; index_active++ )
			{
				for( int index_f = 0; index_f < N_f_mix[ index_measurement ][ index_mix ]; index_f++ )
				{
					index_act = index_active * N_f_mix[ index_measurement ][ index_mix ] + index_f;
					prefactors_mix_float_complex[ index_act ].x = (t_float_gpu) prefactors_mix_complex[ index_act ].real;
					prefactors_mix_float_complex[ index_act ].y = (t_float_gpu) prefactors_mix_complex[ index_act ].imag;
				}
			}
			if( DEBUG_MODE ) mexPrintf( "done!\n" );

			//-------------------------------------------------------------
			// iii.) copy prefactors_mix_float_complex to the device
			//-------------------------------------------------------------
			// allocate memory
			checkCudaErrors( cudaMallocPitch( (void **) &d_prefactors_mix_float_complex, &pitch_prefactors_mix, N_f_mix[ index_measurement ][ index_mix ] * sizeof( t_float_complex_gpu ), N_elements_active_mix[ index_measurement ][ index_mix ] ) );

			// copy data
			checkCudaErrors( cudaMemcpy2D( d_prefactors_mix_float_complex, pitch_prefactors_mix, prefactors_mix_float_complex, N_f_mix[ index_measurement ][ index_mix ] * sizeof( t_float_complex_gpu ), N_f_mix[ index_measurement ][ index_mix ] * sizeof( t_float_complex_gpu ), N_elements_active_mix[ index_measurement ][ index_mix ], cudaMemcpyHostToDevice ) );

			// clean-up host memory
			mxFree( prefactors_mix_float_complex );

			// device memory status
			if( DEBUG_MODE ) printMemInfo();

			//-------------------------------------------------------------
			// iv.) parallelization settings
			//-------------------------------------------------------------
			// number of blocks to process in parallel
			N_blocks_x = ceil( ( (double) N_points_occupied ) / N_THREADS_X );
			N_blocks_y = ceil( ( (double) N_f_mix[ index_measurement ][ index_mix ] ) / N_THREADS_Y );
			dim3 numBlocks( N_blocks_x, N_blocks_y );

			// iterate active array elements
			for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement ][ index_mix ]; index_active++ )
			{

				// index of active array element
				index_element = indices_active_mix[ index_measurement ][ index_mix ][ index_active ];
				if( DEBUG_MODE ) mexPrintf( "index_element = %d\n", index_element );

				//-------------------------------------------------------------
				// compute entries of the observation matrix (N_f_mix[ index_measurement ][ index_mix ] x N_points_occupied)
				//-------------------------------------------------------------
				compute_matrix_kernel<<<numBlocks, threadsPerBlock>>>(
					d_Phi_float_complex, N_f_mix[ index_measurement ][ index_mix ], N_points_occupied,
					d_h_ref_float_complex, pitch_h_ref,
					d_indices_grid_FOV_shift[ index_element ],
					d_indices_f_mix_to_sequence[ index_measurement ][ index_mix ],
					d_p_incident_measurement_float_complex, pitch_p_incident_measurement,
					d_indices_f_mix_to_measurement[ index_measurement ][ index_mix ],
					d_prefactors_mix_float_complex, pitch_prefactors_mix,
					index_active
				);

// TODO: canonical error checking
				// checkCudaErrors( cudaPeekAtLastError() );
				// checkCudaErrors( cudaDeviceSynchronize() );

				//-------------------------------------------------------------
				// compute matrix-matrix product (cuBLAS)
				//-------------------------------------------------------------
				// CUBLAS_OP_N: non-transpose operation / CUBLAS_OP_C: conjugate transpose operation
				if( mode == 1 )
				{

					// forward scattering
					checkCudaErrors(
						cublasCgemm( handle,
							CUBLAS_OP_N, CUBLAS_OP_N,
							N_f_mix[ index_measurement ][ index_mix ], N_objects, N_points_occupied,
							&gemm_alpha, d_Phi_float_complex, N_f_mix[ index_measurement ][ index_mix ], d_gamma_kappa_float_complex, N_points_occupied,
							&gemm_beta, d_u_M_float_complex[ index_measurement ][ index_mix ], N_f_mix[ index_measurement ][ index_mix ]
						)
					);

				}
				else
				{

					// adjoint scattering
					checkCudaErrors(
						cublasCgemm( handle,
							CUBLAS_OP_C, CUBLAS_OP_N,
							N_points_occupied, N_objects, N_f_mix[ index_measurement ][ index_mix ],
							&gemm_alpha, d_Phi_float_complex, N_f_mix[ index_measurement ][ index_mix ], d_u_M_float_complex[ index_measurement ][ index_mix ], N_f_mix[ index_measurement ][ index_mix ],
							&gemm_beta, d_gamma_kappa_float_complex, N_points_occupied
						)
					);

				} // if( mode == 1 )

			} // for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement ][ index_mix ]; index_active++ )

			// clean-up device memory
			checkCudaErrors( cudaFree( d_prefactors_mix_float_complex ) );

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

		// clean-up device memory
		checkCudaErrors( cudaFree( d_p_incident_measurement_float_complex ) );

	} // for index_measurement = 1:numel( operator_born.discretization.spectral )

	// destroy cuBLAS handle
	checkCudaErrors( cublasDestroy( handle ) );

	// clean-up device memory
	checkCudaErrors( cudaFree( d_h_ref_float_complex ) );
	checkCudaErrors( cudaFree( d_Phi_float_complex ) );

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 9.) copy results to the host
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	//---------------------------------------------------------------------
	// a) copy output to the host
	//---------------------------------------------------------------------
	if( mode == 1 )
	{

		// allocate memory
		u_M_float_complex = (t_float_complex_gpu***) mxMalloc( N_measurements * sizeof( t_float_complex_gpu** ) );

		// iterate sequential pulse-echo measurements
		for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
		{

			// allocate memory
			u_M_float_complex[ index_measurement ] = (t_float_complex_gpu**) mxMalloc( N_mix_measurement[ index_measurement ] * sizeof( t_float_complex_gpu* ) );

			// iterate mixed voltage signals
			for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
			{

				// compute size
				size_bytes_u_M_act = N_f_mix[ index_measurement ][ index_mix ] * N_objects * sizeof( t_float_complex_gpu );

				// allocate memory
				u_M_float_complex[ index_measurement ][ index_mix ] = (t_float_complex_gpu*) mxMalloc( size_bytes_u_M_act );

				// copy data
				checkCudaErrors( cudaMemcpy( u_M_float_complex[ index_measurement ][ index_mix ], d_u_M_float_complex[ index_measurement ][ index_mix ], size_bytes_u_M_act, cudaMemcpyDeviceToHost ) );

			} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

		} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
	}
	else
	{

		// allocate memory
		gamma_kappa_float_complex = (t_float_complex_gpu *) mxMalloc( size_bytes_gamma_kappa );

		// copy data
		checkCudaErrors( cudaMemcpy( gamma_kappa_float_complex, d_gamma_kappa_float_complex, size_bytes_gamma_kappa, cudaMemcpyDeviceToHost ) );

		// clean-up device memory
		checkCudaErrors( cudaFree( d_gamma_kappa_float_complex ) );

	} // if( mode == 1 )

	//---------------------------------------------------------------------
	// b) convert output to double
	//---------------------------------------------------------------------
	if( mode == 1 )
	{

		// iterate sequential pulse-echo measurements
		for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
		{

			// iterate mixed voltage signals
			for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
			{

				// iterate objects
				for( int index_object = 0; index_object < N_objects; index_object++ )
				{

					// iterate frequencies
					for( int index_f = 0; index_f < N_f_mix[ index_measurement ][ index_mix ]; index_f++ )
					{

						// compute destination and source indices
						index_act = N_observations_measurement_cs[ index_measurement ] + N_f_mix_cs[ index_measurement ][ index_mix ] + index_object * N_observations + index_f;
						index_src = index_object * N_f_mix[ index_measurement ][ index_mix ] + index_f;

						output_complex[ index_act ].real = (mxDouble) u_M_float_complex[ index_measurement ][ index_mix ][ index_src ].x;
						output_complex[ index_act ].imag = (mxDouble) u_M_float_complex[ index_measurement ][ index_mix ][ index_src ].y;

					} // for( int index_f = 0; index_f < N_f_mix[ index_measurement ][ index_mix ]; index_f++ )

				} // for( int index_object = 0; index_object < N_objects; index_object++ )

			} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

		} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )

	}
	else
	{

		// iterate elements
		if( DEBUG_MODE ) mexPrintf( "converting gamma_kappa_float_complex to double..." );
		for ( int index_object = 0; index_object < N_objects; index_object++ )
		{
			for ( int index_point_occupied = 0; index_point_occupied < N_points_occupied; index_point_occupied++ )
			{
				index_act = index_object * N_points_occupied + index_point_occupied;
				output_complex[ index_act ].real = (mxDouble) gamma_kappa_float_complex[ index_act ].x;
				output_complex[ index_act ].imag = (mxDouble) gamma_kappa_float_complex[ index_act ].y;
			}
		}
		if( DEBUG_MODE ) mexPrintf( "done!\n" );

	} // if( mode == 1 )

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 10.) clean-up memory
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	mxFree( indices_grid_FOV_occupied );

	// input matrix
	mxFree( gamma_kappa_float_complex );

		for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
		{

			for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
			{

				checkCudaErrors( cudaFree( d_u_M_float_complex[ index_measurement ][ index_mix ] ) );
				mxFree( u_M_float_complex[ index_measurement ][ index_mix ] );

			}

			mxFree( d_u_M_float_complex[ index_measurement ] );
			mxFree( u_M_float_complex[ index_measurement ] );

		}

		mxFree( d_u_M_float_complex );
		mxFree( u_M_float_complex );

	for( int index_element = 0; index_element < N_elements; index_element++ )
	{

		checkCudaErrors( cudaFree( d_indices_grid_FOV_shift[ index_element ] ) );
		mxFree( indices_grid_FOV_shift[ index_element ] );

	} // for( int index_element = 0; index_element < N_elements; index_element++ )

	mxFree( d_indices_grid_FOV_shift );
	mxFree( indices_grid_FOV_shift );

	for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
	{

		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
		{

			checkCudaErrors( cudaFree( d_indices_f_mix_to_sequence[ index_measurement ][ index_mix ] ) );
			checkCudaErrors( cudaFree( d_indices_f_mix_to_measurement[ index_measurement ][ index_mix ] ) );

			mxFree( indices_active_mix[ index_measurement ][ index_mix ] );
			mxFree( indices_f_mix_to_sequence[ index_measurement ][ index_mix ] );
			mxFree( indices_f_mix_to_measurement[ index_measurement ][ index_mix ] );

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

		mxFree( d_indices_f_mix_to_sequence[ index_measurement ] );
		mxFree( d_indices_f_mix_to_measurement[ index_measurement ] );

		mxFree( indices_active_mix[ index_measurement ] );
		mxFree( indices_f_mix_to_sequence[ index_measurement ] );
		mxFree( indices_f_mix_to_measurement[ index_measurement ] );

		mxFree( N_f_mix[ index_measurement ] );
		mxFree( N_f_mix_cs[ index_measurement ] );
		mxFree( N_elements_active_mix[ index_measurement ] );

	} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )	

	mxFree( d_indices_f_mix_to_sequence );
	mxFree( d_indices_f_mix_to_measurement );

	mxFree( indices_active_mix );
	mxFree( indices_f_mix_to_sequence );
	mxFree( indices_f_mix_to_measurement );

	mxFree( N_f_mix );
	mxFree( N_f_mix_cs );
	mxFree( N_elements_active_mix );

	mxFree( N_mix_measurement );
	mxFree( N_f_unique_measurement );
	mxFree( N_observations_measurement );
	mxFree( N_observations_measurement_cs );

	// infer and print elapsed time
	// time_elapsed = toc( time_start );
	// mexPrintf( "done! (%f s)\n", time_elapsed );

} // void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// compute entries of the observation matrix (N_f_mix x N_points_occupied)
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void compute_matrix_kernel( t_float_complex_gpu* d_Phi_float_complex, int N_f_mix, int N_points_occupied, t_float_complex_gpu* d_h_ref_float_complex, size_t pitch_h_ref, int* d_indices_grid_FOV_shift, int* d_indices_f_mix_to_sequence, t_float_complex_gpu* d_p_incident_measurement_float_complex, size_t pitch_p_incident_measurement, int* d_indices_f_mix_to_measurement, t_float_complex_gpu* d_prefactors_mix_float_complex, size_t pitch_prefactors_mix, int index_active )
{

	// each thread computes one element in output matrix
	int index_point_occupied = blockIdx.x * blockDim.x + threadIdx.x;
	int index_f = blockIdx.y * blockDim.y + threadIdx.y;
	int index_total = index_point_occupied * N_f_mix + index_f;

// TODO: shared memory for d_indices_grid_FOV_shift possible?

	// multiply matrix entries
	if( index_f < N_f_mix && index_point_occupied < N_points_occupied )
	{

		// compute matrix entry
		d_Phi_float_complex[ index_total ] = cuCmulf(
												cuCmulf(
													*( (t_float_complex_gpu*) ( (char *) d_h_ref_float_complex + d_indices_grid_FOV_shift[ index_point_occupied ] * pitch_h_ref ) + d_indices_f_mix_to_sequence[ index_f ] ),
													*( (t_float_complex_gpu*) ( (char *) d_p_incident_measurement_float_complex + index_point_occupied * pitch_p_incident_measurement ) + d_indices_f_mix_to_measurement[ index_f ] )
												),
												*( (t_float_complex_gpu*) ( (char *) d_prefactors_mix_float_complex + index_active * pitch_prefactors_mix ) + index_f )
											);

	} // if( index_f < N_f_mix && index_point_occupied < N_points_occupied )

} // __global__ void compute_matrix_kernel( t_float_complex_gpu* d_Phi_float_complex, int N_f_mix, int N_points_occupied, t_float_complex_gpu* d_h_ref_float_complex, size_t pitch_h_ref, int* d_indices_grid_FOV_shift, int* d_indices_f_mix_to_sequence, t_float_complex_gpu* d_p_incident_measurement_float_complex, size_t pitch_p_incident_measurement, int* d_indices_f_mix_to_measurement, t_float_complex_gpu* d_prefactors_mix_float_complex, size_t pitch_prefactors_mix, int index_active )

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// check devices and infer status information
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int get_device_info( int index_device, cuInfo* cuda_info )
{

	// check number of devices supporting CUDA
	int N_devices = 0;
	checkCudaErrors( cudaGetDeviceCount( &N_devices ) );
	if( DEBUG_MODE ) mexPrintf( "number of CUDA devices: %d\n", N_devices );

	// if no CUDA-capable device was detected
	if( N_devices < 1 ) mexErrMsgIdAndTxt( "combined_quick_gpu:NoCUDADevices", "Could not find any CUDA-capable device!" );

	// if selected GPU does not exist
	if( DEBUG_MODE ) mexPrintf( "using device: %d\n", index_device );
	if( index_device >= N_devices || index_device < 0 ) mexErrMsgIdAndTxt( "combined_quick_gpu:InvalidCUDADevice", "Invalid device selected!" );
	// assertion: N_devices > 0 && 0 <= index_device < N_devices

	// get properties of selected device
	checkCudaErrors( cudaGetDeviceProperties( &( cuda_info->deviceProp ), index_device ) );

	// get driver version
	checkCudaErrors( cudaDriverGetVersion( &( cuda_info->version_driver ) ) );

	// get runtime version
	checkCudaErrors( cudaRuntimeGetVersion( &( cuda_info->version_runtime ) ) );

} // int get_device_info( int index_device, cuInfo* cuda_info )

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// print device status information
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void print_device_info( int index_device, int N_devices, cudaDeviceProp deviceProp )
{

	mexPrintf( " %s\n", "--------------------------------------------------------------------------------" );
	mexPrintf( " Information for GPU device %-1d of %-1d:\n", index_device, N_devices );
	mexPrintf( " %s\n", "--------------------------------------------------------------------------------" );
	mexPrintf( " %-20s: %-19s", "name of device", deviceProp.name, "" );
    mexPrintf( " %-22s: %-3d\n", "num. multiprocessors", deviceProp.multiProcessorCount );
    mexPrintf( " %-20s: %-1d.%-1d %15s", "compute capability", deviceProp.major, deviceProp.minor, "" );
    //mexPrintf( " %-22s: %-8d\n", "double precision", capability_double );
    mexPrintf( " %-20s: %-6.2f MiByte %4s", "total global memory", ((double) deviceProp.totalGlobalMem) / BYTES_PER_MEBIBYTE, "" );
    mexPrintf( " %-22s: %-6.2f KiByte\n", "total constant memory", ((double) deviceProp.totalConstMem) / BYTES_PER_KIBIBYTE );
    mexPrintf( " %-20s: %-6.2f KiByte %5s", "shared mem. / block", ((double) deviceProp.sharedMemPerBlock) / BYTES_PER_KIBIBYTE, "" );
    mexPrintf( " %-22s: %-8d\n", "registers per block", deviceProp.regsPerBlock );
//     mexPrintf( " %-20s: %2d.%1d %14s", "driver version", driverVersion / 1000, (driverVersion % 100) / 10, "" );
//     mexPrintf( " %-22s: %2d.%1d\n", "runtime version", runtimeVersion / 1000, (runtimeVersion % 100) / 10 );
    mexPrintf( " %-20s: %-8s %10s", "selected precision", "single", "" );
	mexPrintf( " %-22s: %-8d\n", "warp size", deviceProp.warpSize );

}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// print memory information
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void printMemInfo()
{
	// internal variables
	size_t size_bytes_gpu_available = 0;
	size_t size_bytes_gpu_total = 0;

	// get memory status
	checkCudaErrors( cudaMemGetInfo( &size_bytes_gpu_available, &size_bytes_gpu_total ) );

	// print memory status
	mexPrintf( "memory available on GPU: %.2f MiB (%zu B)\n", ( ( double ) size_bytes_gpu_available / BYTES_PER_MEBIBYTE ), size_bytes_gpu_available );
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// clean-up function
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// static void cleanup( void )
// {
// 	mexPrintf("Closing file matlab.data.\n");
// 	checkCudaErrors( cudaFree( d_h_ref_float_complex ) );
// }