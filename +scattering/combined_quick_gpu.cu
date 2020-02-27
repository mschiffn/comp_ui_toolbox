//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// combined_quick_gpu.cu
//
// implementation of
// the forward scattering operator (N_observations_sel x N_points) and
// the adjoint scattering operator (N_points x N_observations_sel) using
// the reference spatial transfer function
//
// 0. output_matrix					mode = 1: u_M mixed voltage signals (N_observations_sel x N_objects)
//									mode = 2: gamma_kappa adjoint relative spatial fluctuations in compressibility (N_points x N_objects)
// = combined_quick_gpu
// (
// 0. operator_born,				object of class scattering.operator_born (scalar)
// 1. mode,							mode of operation (1 = forward, 2 = adjoint)
// 2. input_matrix,					mode = 1: gamma_kappa relative spatial fluctuations in compressibility (N_points x N_objects)
//									mode = 2: u_M mixed voltage signals (N_observations_sel x N_objects)
// )
//
// author: Martin F. Schiffner
// date: 2019-06-29
// modified: 2020-02-26
// All rights reserved!
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// The Thrust template library can do this trivially using thrust::transform, for example:
// thrust::multiplies<thrust::complex<float> > op;
// thrust::transform(thrust::device, x, x + n, y, z, op);
// would iterate over each pair of inputs from the device pointers x and y and calculate z[i] = x[i] * y[i]
// (there is probably a couple of casts you need to make to compile that, but you get the idea). But that effectively requires compilation of CUDA code within your project, and apparently you don't want that.

// TODO: also use pinned buffer memory for real values and integers!
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
#define REVISION "0.2"
#define DATE "2019-09-05"

// toggle debug mode
#define DEBUG_MODE 0
// #define VERBOSITY 3

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// MEX gateway function
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )
{

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 1.) define local variables
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	//------------------------------------------------------------------------------------------------------------------------------------------
	// a) host variables
	//------------------------------------------------------------------------------------------------------------------------------------------
	// spatial discretization
	int N_points = 0;									// number of grid points
	int N_elements = 0;									// number of array elements
	int N_objects = 0;									// number of objects

	// occupied grid points
	int N_points_occupied = 0;							// number of occupied grid points
	int* indices_grid_FOV_occupied = NULL;				// indices of occupied grid points

	// grid point maps
	int* indices_grid_FOV_shift = NULL;					// indices of shifted grid points

	// spectral discretizations
	int N_measurements_sel = 0;							// number of selected sequential pulse-echo measurements
	int index_measurement = 0;							// index of sequential pulse-echo measurement
	mxDouble* indices_measurement_sel_double = NULL;
	int N_f_unique = 0;									// number of unique frequencies
	int* N_f_unique_measurement = 0;					// number of unique frequencies in each pulse-echo measurement
	int* N_mix_measurement = NULL;						// number of mixed voltage signals in each pulse-echo measurement
	int** N_f_mix = NULL;								// number of frequencies in each mixed voltage signal
	int** N_f_mix_cs = NULL;							// number of frequencies in each mixed voltage signal (cumulative sum)
	int** N_elements_active_mix = NULL;					// number of active array elements in each mixed voltage signal
	int*** indices_active_mix = NULL;					// indices of active array elements in each mixed voltage signal
	mxDouble* indices_active_mix_double = NULL;
	int N_observations_sel = 0;							// number of selected observations
	int* N_observations_measurement = NULL;				// number of observations in each pulse-echo measurement
	int* N_observations_measurement_cs = NULL;			// number of observations in each pulse-echo measurement (cumulative sum)

	// statistics
	int N_f_unique_measurement_max = 0;					// maximum number of unique frequencies in all pulse-echo measurements
	int N_f_mix_max = 0;								// maximum number of frequencies in all mixed voltage signals

	// frequency maps
	int*** indices_f_mix_to_measurement = NULL;			// indices of unique frequencies in each pulse-echo measurement for each mixed voltage signal
	int*** indices_f_mix_to_sequence = NULL;			// indices of unique frequencies for each mixed voltage signal

	// options
	int mode = 1;										// mode of operation (1 = forward, other = adjoint)
	int index_device = 0;								// index of CUDA device

	// dimensions of output vector
	int N_columns = 0;									// number of columns in the operator
	mwSize dimensions_output[ 2 ];

	// reference spatial transfer function
	mxArray* h_ref = NULL;
	mxComplexDouble* h_ref_complex = NULL;

	// page-locked buffer for conversion to float
	size_t size_bytes_buffer = 0;
	t_float_complex_gpu* buffer_float_complex = NULL;

	// indices of shifted grid points
	mxDouble* indices_grid_FOV_shift_double = NULL;

	// indices of unique frequencies
	mxDouble* indices_f_measurement_to_sequence_double = NULL;
	mxDouble* indices_f_mix_to_measurement_double = NULL;

	// incident acoustic pressure field
	mxArray* p_incident_measurement = NULL;
	mxComplexDouble * p_incident_measurement_complex = NULL;

	size_t size_bytes_p_incident_max = 0;
	size_t size_bytes_p_incident_measurement = 0;

	// prefactors for current mix
	mxComplexDouble*** prefactors_mix_complex = NULL;

	// input matrix (compressibility fluctuations / mixed voltage signals)
	mxArray* input_conversion = NULL;
	mxComplexDouble* input_complex = NULL;

	// output matrix (mixed voltage signals / adjoint compressibility fluctuations)
	mxComplexDouble* output_complex = NULL;

	// misc variables
	size_t size_bytes_1 = 0, size_bytes_2 = 0;
	mxDouble* size = NULL;
	mxArray* sequence = NULL;
	mxArray* sequence_settings = NULL;
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

	//------------------------------------------------------------------------------------------------------------------------------------------
	// b) device variables
	//------------------------------------------------------------------------------------------------------------------------------------------
	size_t pitch_h_ref = 0;
	t_float_complex_gpu* d_h_ref_float_complex = NULL;
	size_t pitch_p_incident_measurement = 0;
	t_float_complex_gpu* d_p_incident_measurement_float_complex = NULL;
	t_float_complex_gpu**** d_prefactors_mix_float_complex = NULL;

	// input and output matrices
	size_t size_bytes_gamma_kappa = 0;
	t_float_complex_gpu* d_gamma_kappa_float_complex = NULL;		// relative spatial fluctuations in compressibility
	size_t size_bytes_u_M_act = 0;
	t_float_complex_gpu*** d_u_M_float_complex = NULL;				// mixed voltage signals

	// intermediate results
	t_float_complex_gpu* d_Phi_float_complex = NULL;

	int** d_indices_grid_FOV_shift = NULL;							// indices of shifted grid points for each array element
	int*** d_indices_f_mix_to_measurement = NULL;					// frequency indices in each pulse-echo measurement
	int*** d_indices_f_mix_to_sequence = NULL;						// frequency indices in each mixed voltage signal

	size_t size_bytes_indices_f_act = 0;
	size_t size_bytes_Phi_max = 0;

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 2.) check arguments
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// check number of arguments and outputs
	if ( nrhs != 3 || nlhs != 1 ) mexErrMsgIdAndTxt( "combined_quick_gpu:NumberArguments", "combined_quick_gpu requires 3 inputs and 1 output." );

	//------------------------------------------------------------------------------------------------------------------------------------------
	// a) ensure class scattering.operator_born (scalar)
	//------------------------------------------------------------------------------------------------------------------------------------------
	// ensure class scattering.operator_born (scalar)
	if( mxIsClass( prhs[ 0 ], "scattering.operator_born" ) && mxIsScalar( prhs[ 0 ] ) )
	{

		sequence = mxGetProperty( prhs[ 0 ], 0, "sequence" );
		size = mxGetDoubles( mxGetProperty( sequence, 0, "size" ) );

		sequence_settings = mxGetProperty( sequence, 0, "settings" );

		// number of observations
		// N_observations_sel = mxGetM( prhs[ 2 ] );
		N_observations_sel = (int) size[ 0 ];
		if( DEBUG_MODE ) mexPrintf( "N_observations_sel = %d\n", N_observations_sel );

		// number of grid points
		// N_points = (int) mxGetScalar( mxGetProperty( mxGetProperty( mxGetProperty( sequence, 0, "setup" ), 0, "grid_FOV" ), 0, "N_points" ) );
		N_points = (int) size[ 1 ];
		if( DEBUG_MODE ) mexPrintf( "N_points = %d\n", N_points );

		// ensure class scattering.sequences.setups.setup_grid_symmetric
		if( !mxIsClass( mxGetProperty( sequence, 0, "setup" ), "scattering.sequences.setups.setup_grid_symmetric" ) ) mexErrMsgIdAndTxt( "combined_quick_gpu:NoSymmetricGridsSetup", "sequence.setup must be scattering.sequences.setups.setup_grid_symmetric!" );

		// extract indices of shifted grid points
		indices_grid_FOV_shift_double = mxGetDoubles( mxGetProperty( mxGetProperty( sequence, 0, "setup" ), 0, "indices_grid_FOV_shift" ) );

		// number of array elements
		N_elements = (int) mxGetScalar( mxGetProperty( mxGetProperty( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "sequence" ), 0, "setup" ), 0, "xdc_array" ), 0, "N_elements" ) );
		if( DEBUG_MODE ) mexPrintf( "N_elements = %d\n", N_elements );

		// extract indices of selected sequential pulse-echo measurements
		indices_measurement_sel_double = mxGetDoubles( mxGetProperty( prhs[ 0 ], 0, "indices_measurement_sel" ) );

		// number of selected sequential pulse-echo measurements
		N_measurements_sel = mxGetNumberOfElements( mxGetProperty( prhs[ 0 ], 0, "indices_measurement_sel" ) );
		if( DEBUG_MODE ) mexPrintf( "N_measurements_sel = %d\n", N_measurements_sel );

		// allocate memory
		N_mix_measurement = (int*) mxMalloc( N_measurements_sel * sizeof( int ) );
		N_f_unique_measurement = (int*) mxMalloc( N_measurements_sel * sizeof( int ) );
		N_f_mix = (int**) mxMalloc( N_measurements_sel * sizeof( int* ) );
		N_f_mix_cs = (int**) mxMalloc( N_measurements_sel * sizeof( int* ) );
		N_elements_active_mix = (int**) mxMalloc( N_measurements_sel * sizeof( int* ) );
		N_observations_measurement = (int*) mxMalloc( N_measurements_sel * sizeof( int ) );
		N_observations_measurement_cs = (int*) mxCalloc( N_measurements_sel, sizeof( int ) );
		indices_f_mix_to_measurement = (int***) mxMalloc( N_measurements_sel * sizeof( int** ) );
		indices_f_mix_to_sequence = (int***) mxMalloc( N_measurements_sel * sizeof( int** ) );
		indices_active_mix = (int***) mxMalloc( N_measurements_sel * sizeof( int** ) );
		prefactors_mix_complex = (mxComplexDouble***) mxMalloc( N_measurements_sel * sizeof( mxComplexDouble** ) );

		// iterate selected sequential pulse-echo measurements
		for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )
		{

			// index of sequential pulse-echo measurement
			index_measurement = (int) indices_measurement_sel_double[ index_measurement_sel ] - 1;

			// numbers of mixed voltage signals per measurement
			N_mix_measurement[ index_measurement_sel ] = mxGetNumberOfElements( mxGetProperty( sequence_settings, index_measurement, "rx" ) );
			if( DEBUG_MODE ) mexPrintf( "N_mix_measurement[%d] = %d\n", index_measurement_sel, N_mix_measurement[ index_measurement_sel ] );

			// allocate memory
			N_f_mix[ index_measurement_sel ] = (int*) mxMalloc( N_mix_measurement[ index_measurement_sel ] * sizeof( int ) );
			N_f_mix_cs[ index_measurement_sel ] = (int*) mxCalloc( N_mix_measurement[ index_measurement_sel ], sizeof( int ) );
			N_elements_active_mix[ index_measurement_sel ] = (int*) mxMalloc( N_mix_measurement[ index_measurement_sel ] * sizeof( int ) );
			indices_f_mix_to_measurement[ index_measurement_sel ] = (int**) mxMalloc( N_mix_measurement[ index_measurement_sel ] * sizeof( int* ) );
			indices_f_mix_to_sequence[ index_measurement_sel ] = (int**) mxMalloc( N_mix_measurement[ index_measurement_sel ] * sizeof( int* ) );
			indices_active_mix[ index_measurement_sel ] = (int**) mxMalloc( N_mix_measurement[ index_measurement_sel ] * sizeof( int* ) );
			prefactors_mix_complex[ index_measurement_sel ] = (mxComplexDouble**) mxMalloc( N_mix_measurement[ index_measurement_sel ] * sizeof( mxComplexDouble* ) );

			// map unique frequencies of pulse-echo measurement to global unique frequencies
			indices_f_measurement_to_sequence_double = mxGetDoubles( mxGetCell( mxGetProperty( sequence, 0, "indices_f_to_unique" ), (mwIndex) index_measurement ) );

			// number of unique frequencies in current measurement
			N_f_unique_measurement[ index_measurement_sel ] = mxGetM( mxGetCell( mxGetProperty( sequence, 0, "indices_f_to_unique" ), (mwIndex) index_measurement ) );
			if( DEBUG_MODE ) mexPrintf( "N_f_unique_measurement[%d] = %d\n", index_measurement_sel, N_f_unique_measurement[ index_measurement_sel ] );

			// maximum number of unique frequencies in all pulse-echo measurements
			if( N_f_unique_measurement[ index_measurement_sel ] > N_f_unique_measurement_max ) N_f_unique_measurement_max = N_f_unique_measurement[ index_measurement_sel ];

			// number of observations in each pulse-echo measurement
			N_observations_measurement[ index_measurement_sel ] = 0;

			// extract prefactors for all mixes
			prefactors_measurement = mxGetCell( mxGetProperty( sequence, 0, "prefactors" ), (mwIndex) index_measurement );

			// iterate mixed voltage signals
			for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )
			{

				// number of frequencies in current mix
				N_f_mix[ index_measurement_sel ][ index_mix ] = mxGetM( mxGetProperty( mxGetProperty( mxGetCell( mxGetProperty( sequence, 0, "prefactors" ), (mwIndex) index_measurement ), index_mix, "samples" ), 0, "values" ) );
				if( DEBUG_MODE ) mexPrintf( "N_f_mix[%d][%d] = %d\n", index_measurement_sel, index_mix, N_f_mix[ index_measurement_sel ][ index_mix ] );

				// maximum number of frequencies in all mixed voltage signals
				if( N_f_mix[ index_measurement_sel ][ index_mix ] > N_f_mix_max ) N_f_mix_max = N_f_mix[ index_measurement_sel ][ index_mix ];

				// number of frequencies in each mixed voltage signal (cumulative sum)
				if( index_mix > 0 ) N_f_mix_cs[ index_measurement_sel ][ index_mix ] = N_f_mix_cs[ index_measurement_sel ][ index_mix - 1 ] + N_f_mix[ index_measurement_sel ][ index_mix - 1 ];
				if( DEBUG_MODE ) mexPrintf( "N_f_mix_cs[%d][%d] = %d\n", index_measurement_sel, index_mix, N_f_mix_cs[ index_measurement_sel ][ index_mix ] );

				// number of active array elements in each mixed voltage signal
				rx_measurement = mxGetProperty( mxGetProperty( sequence_settings, index_measurement, "rx" ), index_mix, "indices_active" );
				N_elements_active_mix[ index_measurement_sel ][ index_mix ] = mxGetNumberOfElements( rx_measurement );
				if( DEBUG_MODE ) mexPrintf( "N_elements_active_mix[%d][%d] = %d\n", index_measurement_sel, index_mix, N_elements_active_mix[ index_measurement_sel ][ index_mix ] );

				// indices of active array elements for each mixed voltage signal
				indices_active_mix_double = mxGetDoubles( rx_measurement );

				// map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
				indices_f_mix_to_measurement_double = mxGetDoubles( mxGetCell( mxGetProperty( sequence_settings, index_measurement, "indices_f_to_unique" ), (mwIndex) index_mix ) );

				// allocate memory
				indices_f_mix_to_measurement[ index_measurement_sel ][ index_mix ] = (int*) mxMalloc( N_f_mix[ index_measurement_sel ][ index_mix ] * sizeof( int ) );
				indices_f_mix_to_sequence[ index_measurement_sel ][ index_mix ] = (int*) mxMalloc( N_f_mix[ index_measurement_sel ][ index_mix ] * sizeof( int ) );
				indices_active_mix[ index_measurement_sel ][ index_mix ] = (int*) mxMalloc( N_elements_active_mix[ index_measurement_sel ][ index_mix ] * sizeof( int ) );

				// iterate frequencies
				for( int index_f = 0; index_f < N_f_mix[ index_measurement_sel ][ index_mix ]; index_f++ )
				{

					// indices of unique frequencies in each pulse-echo measurement for each mixed voltage signal
					indices_f_mix_to_measurement[ index_measurement_sel ][ index_mix ][ index_f ] = (int) indices_f_mix_to_measurement_double[ index_f ] - 1;
					if( DEBUG_MODE ) mexPrintf( "indices_f_mix_to_measurement[%d][%d][%d] = %d\n", index_measurement_sel, index_mix, index_f, indices_f_mix_to_measurement[ index_measurement_sel ][ index_mix ][ index_f ] );

					// indices of unique frequencies for each mixed voltage signal
					indices_f_mix_to_sequence[ index_measurement_sel ][ index_mix ][ index_f ] = (int) indices_f_measurement_to_sequence_double[ indices_f_mix_to_measurement[ index_measurement_sel ][ index_mix ][ index_f ] ] - 1;
					if( DEBUG_MODE ) mexPrintf( "indices_f_mix_to_sequence[%d][%d][%d] = %d\n", index_measurement_sel, index_mix, index_f, indices_f_mix_to_sequence[ index_measurement_sel ][ index_mix ][ index_f ] );

				} // for( int index_f = 0; index_f < N_f_mix[ index_measurement_sel ][ index_mix ]; index_f++ )

				// iterate active array elements
				for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement_sel ][ index_mix ]; index_active++ )
				{

					// indices of active array elements for each mixed voltage signal
					indices_active_mix[ index_measurement_sel ][ index_mix ][ index_active ] = (int) indices_active_mix_double[ index_active ] - 1;
					if( DEBUG_MODE ) mexPrintf( "indices_active_mix[%d][%d][%d] = %d\n", index_measurement_sel, index_mix, index_active, indices_active_mix[ index_measurement_sel ][ index_mix ][ index_active ] );

				} // for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement_sel ][ index_mix ]; index_active++ )

				// number of observations in each pulse-echo measurement
				N_observations_measurement[ index_measurement_sel ] += N_f_mix[ index_measurement_sel ][ index_mix ];

				// extract prefactors_mix_complex
				prefactors_mix_complex[ index_measurement_sel ][ index_mix ] = mxGetComplexDoubles( mxGetProperty( mxGetProperty( prefactors_measurement, index_mix, "samples" ), 0, "values" ) );

			} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )

			if( index_measurement_sel > 0 ) N_observations_measurement_cs[ index_measurement_sel ] = N_observations_measurement_cs[ index_measurement_sel - 1 ] + N_observations_measurement[ index_measurement_sel - 1 ];

			if( DEBUG_MODE )
			{
				mexPrintf( "N_observations_measurement[%d] = %d\n", index_measurement_sel, N_observations_measurement[ index_measurement_sel ] );
				mexPrintf( "N_observations_measurement_cs[%d] = %d\n", index_measurement_sel, N_observations_measurement_cs[ index_measurement_sel ] );
				mexPrintf( "N_f_unique_measurement_max = %d\n", N_f_unique_measurement_max );
				mexPrintf( "N_f_mix_max = %d\n", N_f_mix_max );
			}

		} // for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )

		// number of selected observations
		N_observations_sel = N_observations_measurement_cs[ N_measurements_sel - 1 ] + N_observations_measurement[ N_measurements_sel - 1 ];

		// ensure class scattering.options.gpu_active
		if( !mxIsClass( mxGetProperty( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "options" ), 0, "momentary" ), 0, "gpu" ), "scattering.options.gpu_active" ) )
		{
			mexErrMsgIdAndTxt( "combined_quick_gpu:NoScatteringOptionsGPUActive", "options.momentary.gpu must be scattering.options.gpu_active!" );
		}

		// index of CUDA device
		index_device = (int) mxGetScalar( mxGetProperty( mxGetProperty( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "options" ), 0, "momentary" ), 0, "gpu" ), 0, "index" ) );
		if( DEBUG_MODE ) mexPrintf( "index_device = %d\n", index_device );

		// extract reference spatial transfer function
		h_ref = mxGetProperty( mxGetProperty( mxGetProperty( sequence, 0, "h_ref" ), 0, "samples" ), 0, "values" );

		// ensure complex doubles (mxDOUBLE_CLASS)
		if( !( mxIsDouble( h_ref ) && mxIsComplex( h_ref ) ) ) mexErrMsgIdAndTxt( "combined_quick_gpu:NoComplexDoubles", "operator_born.sequence.h_ref must be complex doubles!" );

		// read complex doubles
		h_ref_complex = mxGetComplexDoubles( h_ref );

		// number of unique frequencies
		N_f_unique = mxGetM( h_ref );
		if( DEBUG_MODE ) mexPrintf( "N_f_unique = %d\n", N_f_unique );

	}
	else
	{
		mexErrMsgIdAndTxt( "combined_quick_gpu:NoOperatorBorn", "operator_born must be a single scattering.operator_born!" );
	} // if( mxIsClass( prhs[ 0 ], "scattering.operator_born" ) && mxIsScalar( prhs[ 0 ] ) )

	//------------------------------------------------------------------------------------------------------------------------------------------
	// b) mode of operation
	//------------------------------------------------------------------------------------------------------------------------------------------
	// TODO: Use mxGetScalar on a nonempty mxArray of type numeric, logical, or char only. mxIsEmpty, mxIsLogical, mxIsNumeric, or mxIsChar
	if( mxIsNumeric( prhs[ 1 ] ) && mxIsScalar( prhs[ 1 ] ) && !mxIsComplex( prhs[ 1 ] ) )
	{

		mode = (int) mxGetScalar( prhs[ 1 ] );
		if( DEBUG_MODE ) mexPrintf( "mode = %d\n", mode );
		N_columns = ( mode == 1 ) ? N_points : N_observations_sel;

	}
	else
	{
		mexErrMsgIdAndTxt( "combined_quick_gpu:NoNumericScalarMode", "mode must be a real-valued numeric scalar!" );
	}

	//------------------------------------------------------------------------------------------------------------------------------------------
	// c) input matrix (N_columns x N_objects)
	//------------------------------------------------------------------------------------------------------------------------------------------
	// ensure nonempty matrix of doubles (mxDOUBLE_CLASS) w/ correct number of rows
	if( !( mxIsDouble( prhs[ 2 ] ) && mxGetNumberOfDimensions( prhs[ 2 ] ) == 2 && mxGetM( prhs[ 2 ] ) == N_columns && !mxIsEmpty( prhs[ 2 ] ) ) ) mexErrMsgIdAndTxt( "combined_quick_gpu:NoValidDoubleMatrix", "input must be a double matrix with xxx rows!" );

	// ensure imaginary part for input_conversion
	if( !mxIsComplex( prhs[ 2 ] ) )
	{
		// deep copy constant input array
		input_conversion = mxDuplicateArray( prhs[ 2 ] );
		if( !mxMakeArrayComplex( input_conversion ) ) mexErrMsgIdAndTxt( "combined_quick_gpu:ConversionFailed", "Could not convert real mxArray input_conversion to complex, preserving real data!" );

		// extract input matrix
		input_complex = mxGetComplexDoubles( input_conversion );
	}
	else
	{
		// extract input matrix
		input_complex = mxGetComplexDoubles( prhs[ 2 ] );
	}

	// number of objects
	N_objects = mxGetN( prhs[ 2 ] );
	if( DEBUG_MODE ) mexPrintf( "N_objects = %d\n", N_objects );

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 3.) MATLAB output
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// allocate workspace memory
	dimensions_output[ 0 ] = ( mode == 1 ) ? N_observations_sel : N_points;
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
	// 5.)  allocate and initialize device memory / convert data to float and copy to the device
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// TODO: also use buffer for real values and integers!
	// size of conversion buffer: use maximum of
		// a) d_h_ref_float_complex:
		//	  N_f_unique * N_points * sizeof( t_float_complex_gpu ) [ reference spatial transfer function ] (checked!)
		// b) d_prefactors_mix_float_complex[ index_measurement_sel ][ index_mix ][ index_active ]:
		//    N_f_mix_max * sizeof( t_float_complex_gpu ) [ prefactors ]
		// c) d_gamma_kappa_float_complex:
		//    N_points_occupied * N_objects * sizeof( t_float_complex_gpu ) [ relative spatial fluctuations in compressibility ]
		// d) d_u_M_float_complex[ index_measurement_sel ][ index_mix ]:
		//	  N_f_mix_max * N_objects * sizeof( t_float_complex_gpu )
		// e) d_p_incident_measurement_float_complex:
		//	  N_f_unique_measurement_max * N_points_occupied * sizeof( t_float_complex_gpu )
		//
		// => a) > b), e)
		// => a), d) vs. c)
		// N_f_unique * max( N_points, N_objects ) vs. N_points_occupied * N_objects

		// b) d_indices_grid_FOV_shift[ index_element ]: N_points_occupied * sizeof( int ) [ indices of shifted grid points for each array element ]

	//size_bytes_buffer = N_f_unique * N_points * sizeof( t_float_complex_gpu );	
	size_bytes_1 = N_f_unique * ( ( N_points > N_objects ) ? N_points : N_objects );
	size_bytes_2 = N_points_occupied * N_objects;
	size_bytes_buffer = ( ( size_bytes_1 > size_bytes_2 ) ? size_bytes_1 : size_bytes_2 ) * sizeof( t_float_complex_gpu );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_buffer = %.2f MiB (%zu B)\n", ( ( double ) size_bytes_buffer ) / BYTES_PER_MEBIBYTE, size_bytes_buffer );

	// allocate page-locked memory
	checkCudaErrors( cudaHostAlloc( (void**) &buffer_float_complex, size_bytes_buffer, cudaHostAllocDefault ) );

	//------------------------------------------------------------------------------------------------------------------------------------------
	// a) reference spatial transfer function (N_f_unique x N_points)
	//------------------------------------------------------------------------------------------------------------------------------------------
	// iterate matrix entries
	if( DEBUG_MODE ) mexPrintf( "converting h_ref_complex to float... " );
	for( int index_entry = 0; index_entry < N_f_unique * N_points; index_entry++ )
	{
		buffer_float_complex[ index_entry ].x = (t_float_gpu) h_ref_complex[ index_entry ].real;
		buffer_float_complex[ index_entry ].y = (t_float_gpu) h_ref_complex[ index_entry ].imag;
	}
	if( DEBUG_MODE ) mexPrintf( "done!\n" );

	// allocate device memory
	checkCudaErrors( cudaMallocPitch( (void **) &d_h_ref_float_complex, &pitch_h_ref, N_f_unique * sizeof( t_float_complex_gpu ), N_points ) );

	// copy data
	checkCudaErrors( cudaMemcpy2D( d_h_ref_float_complex, pitch_h_ref, buffer_float_complex, N_f_unique * sizeof( t_float_complex_gpu ), N_f_unique * sizeof( t_float_complex_gpu ), N_points, cudaMemcpyHostToDevice ) );

	// register an exit function
	// mexAtExit( cleanup );

	//------------------------------------------------------------------------------------------------------------------------------------------
	// b) indices of shifted grid points for each array element (N_points_occupied x N_elements)
	//------------------------------------------------------------------------------------------------------------------------------------------
	// allocate memory
	indices_grid_FOV_shift = (int*) mxMalloc( N_points_occupied * sizeof( int ) );
	d_indices_grid_FOV_shift = (int**) mxMalloc( N_elements * sizeof( int* ) );

	// iterate array elements
	if( DEBUG_MODE ) mexPrintf( "converting indices_grid_FOV_shift_double to int... " );
	for( int index_element = 0; index_element < N_elements; index_element++ )
	{

		// iterate occupied grid points
		for( int index_point_occupied = 0; index_point_occupied < N_points_occupied; index_point_occupied++ )
		{
			index_src = index_element * N_points + indices_grid_FOV_occupied[ index_point_occupied ];
			indices_grid_FOV_shift[ index_point_occupied ] = (int) indices_grid_FOV_shift_double[ index_src ] - 1;
		}

		// allocate device memory
		checkCudaErrors( cudaMalloc( (void**) &( d_indices_grid_FOV_shift[ index_element ] ), N_points_occupied * sizeof( int ) ) );

		// copy data
		checkCudaErrors( cudaMemcpy( d_indices_grid_FOV_shift[ index_element ], indices_grid_FOV_shift, N_points_occupied * sizeof( int ), cudaMemcpyHostToDevice ) );

	} // for( int index_element = 0; index_element < N_elements; index_element++ )
	if( DEBUG_MODE ) mexPrintf( "done!\n" );

	// clean-up host memory
	mxFree( indices_grid_FOV_shift );

	//------------------------------------------------------------------------------------------------------------------------------------------
	// c) copy indices_f_mix_to_measurement and indices_f_mix_to_sequence to the device
	//------------------------------------------------------------------------------------------------------------------------------------------
	// allocate memory
	d_indices_f_mix_to_measurement = (int***) mxMalloc( N_measurements_sel * sizeof( int** ) );
	d_indices_f_mix_to_sequence = (int***) mxMalloc( N_measurements_sel * sizeof( int** ) );

	// iterate selected sequential pulse-echo measurements
	for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )
	{

		// allocate memory
		d_indices_f_mix_to_measurement[ index_measurement_sel ] = (int**) mxMalloc( N_mix_measurement[ index_measurement_sel ] * sizeof( t_float_complex_gpu* ) );
		d_indices_f_mix_to_sequence[ index_measurement_sel ] = (int**) mxMalloc( N_mix_measurement[ index_measurement_sel ] * sizeof( t_float_complex_gpu* ) );

		// iterate mixed voltage signals
		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )
		{

			// compute size
			size_bytes_indices_f_act = N_f_mix[ index_measurement_sel ][ index_mix ] * sizeof( int );

			// allocate device memory
			checkCudaErrors( cudaMalloc( (void **) &( d_indices_f_mix_to_measurement[ index_measurement_sel ][ index_mix ] ), size_bytes_indices_f_act ) );
			checkCudaErrors( cudaMalloc( (void **) &( d_indices_f_mix_to_sequence[ index_measurement_sel ][ index_mix ] ), size_bytes_indices_f_act ) );

			// copy data
			checkCudaErrors( cudaMemcpy( d_indices_f_mix_to_measurement[ index_measurement_sel ][ index_mix ], indices_f_mix_to_measurement[ index_measurement_sel ][ index_mix ], size_bytes_indices_f_act, cudaMemcpyHostToDevice ) );
			checkCudaErrors( cudaMemcpy( d_indices_f_mix_to_sequence[ index_measurement_sel ][ index_mix ], indices_f_mix_to_sequence[ index_measurement_sel ][ index_mix ], size_bytes_indices_f_act, cudaMemcpyHostToDevice ) );

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )

	} // for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )

	//------------------------------------------------------------------------------------------------------------------------------------------
	// d) allocate memory for d_p_incident_measurement_float_complex (use maximum size: N_f_unique_measurement_max x N_points_occupied)
	//------------------------------------------------------------------------------------------------------------------------------------------
	// compute size
	size_bytes_p_incident_max = N_f_unique_measurement_max * N_points_occupied * sizeof( t_float_complex_gpu );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_p_incident_max = %.2f MiB (%zu B)\n", ( (double) size_bytes_p_incident_max ) / BYTES_PER_MEBIBYTE, size_bytes_p_incident_max );

	// allocate device memory
	checkCudaErrors( cudaMallocPitch( (void **) &d_p_incident_measurement_float_complex, &pitch_p_incident_measurement, N_f_unique_measurement_max * sizeof( t_float_complex_gpu ), N_points_occupied ) );

	//------------------------------------------------------------------------------------------------------------------------------------------
	// e) convert prefactors_mix_complex to float
	//------------------------------------------------------------------------------------------------------------------------------------------
	// allocate memory
	d_prefactors_mix_float_complex = (t_float_complex_gpu****) mxMalloc( N_measurements_sel * sizeof( t_float_complex_gpu*** ) );

	// iterate selected sequential pulse-echo measurements
	if( DEBUG_MODE ) mexPrintf( "converting prefactors_mix_complex to float... " );
	for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )
	{

		// allocate memory
		d_prefactors_mix_float_complex[ index_measurement_sel ] = (t_float_complex_gpu***) mxMalloc( N_mix_measurement[ index_measurement_sel ] * sizeof( t_float_complex_gpu** ) );

		// iterate mixed voltage signals
		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )
		{

			// allocate memory
			d_prefactors_mix_float_complex[ index_measurement_sel ][ index_mix ] = (t_float_complex_gpu**) mxMalloc( N_elements_active_mix[ index_measurement_sel ][ index_mix ] * sizeof( t_float_complex_gpu* ) );

			// iterate active array elements
			for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement_sel ][ index_mix ]; index_active++ )
			{

				// iterate frequencies
				for( int index_f = 0; index_f < N_f_mix[ index_measurement_sel ][ index_mix ]; index_f++ )
				{
					index_act = index_active * N_f_mix[ index_measurement_sel ][ index_mix ] + index_f;
					buffer_float_complex[ index_f ].x = (t_float_gpu) prefactors_mix_complex[ index_measurement_sel ][ index_mix ][ index_act ].real;
					buffer_float_complex[ index_f ].y = (t_float_gpu) prefactors_mix_complex[ index_measurement_sel ][ index_mix ][ index_act ].imag;
				}

				// allocate device memory
				checkCudaErrors( cudaMalloc( &( d_prefactors_mix_float_complex[ index_measurement_sel ][ index_mix ][ index_active ] ), N_f_mix[ index_measurement_sel ][ index_mix ] * sizeof( t_float_complex_gpu ) ) );

				// copy data
				checkCudaErrors( cudaMemcpy( d_prefactors_mix_float_complex[ index_measurement_sel ][ index_mix ][ index_active ], buffer_float_complex, N_f_mix[ index_measurement_sel ][ index_mix ] * sizeof( t_float_complex_gpu ), cudaMemcpyHostToDevice ) );

			} // for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement_sel ][ index_mix ]; index_active++ )

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )

	} // for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )
	if( DEBUG_MODE ) mexPrintf( "done!\n" );

	//------------------------------------------------------------------------------------------------------------------------------------------
	// f) d_Phi_float_complex (use maximum size: N_f_mix_max x N_points_occupied)
	//------------------------------------------------------------------------------------------------------------------------------------------
	// compute maximum size
	size_bytes_Phi_max = N_f_mix_max * N_points_occupied * sizeof( t_float_complex_gpu );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_Phi_max = %.2f MiB (%zu B)\n", ( (double) size_bytes_Phi_max ) / BYTES_PER_MEBIBYTE, size_bytes_Phi_max );

	// allocate device memory
	checkCudaErrors( cudaMalloc( (void **) &d_Phi_float_complex, size_bytes_Phi_max ) );

	//------------------------------------------------------------------------------------------------------------------------------------------
	// g) relative spatial fluctuations in compressibility (N_points_occupied x N_objects)
	//------------------------------------------------------------------------------------------------------------------------------------------
	// compute size
	size_bytes_gamma_kappa = N_points_occupied * N_objects * sizeof( t_float_complex_gpu );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_gamma_kappa = %.2f MiB (%zu B)\n", ( ( double ) size_bytes_gamma_kappa ) / BYTES_PER_MEBIBYTE, size_bytes_gamma_kappa );

	// allocate device memory
	checkCudaErrors( cudaMalloc( (void **) &d_gamma_kappa_float_complex, size_bytes_gamma_kappa ) );

	// initialize device memory
	if( mode == 1)
	{
		// iterate matrix entries
		if( DEBUG_MODE ) mexPrintf( "converting input_complex to float... " );
		for( int index_object = 0; index_object < N_objects; index_object++ )
		{
			for( int index_point_occupied = 0; index_point_occupied < N_points_occupied; index_point_occupied++ )
			{
				index_act = index_object * N_points_occupied + index_point_occupied;
				index_src = index_object * N_points + indices_grid_FOV_occupied[ index_point_occupied ];
				buffer_float_complex[ index_act ].x = (t_float_gpu) input_complex[ index_src ].real;
				buffer_float_complex[ index_act ].y = (t_float_gpu) input_complex[ index_src ].imag;
			}
		}
		if( DEBUG_MODE ) mexPrintf( "done!\n" );

		// copy data
		checkCudaErrors( cudaMemcpy( d_gamma_kappa_float_complex, buffer_float_complex, size_bytes_gamma_kappa, cudaMemcpyHostToDevice ) );
	}
	else
	{
		// set device memory to zero
		checkCudaErrors( cudaMemset( d_gamma_kappa_float_complex, 0, size_bytes_gamma_kappa ) );
	}

	//------------------------------------------------------------------------------------------------------------------------------------------
	// h) mixed voltage signals (N_observations_sel x N_objects)
	//------------------------------------------------------------------------------------------------------------------------------------------
	// allocate memory
	d_u_M_float_complex = (t_float_complex_gpu***) mxMalloc( N_measurements_sel * sizeof( t_float_complex_gpu** ) );

	// iterate selected sequential pulse-echo measurements
	for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )
	{
		// allocate memory
		d_u_M_float_complex[ index_measurement_sel ] = (t_float_complex_gpu**) mxMalloc( N_mix_measurement[ index_measurement_sel ] * sizeof( t_float_complex_gpu* ) );

		// iterate mixed voltage signals
		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )
		{
			// compute size
			size_bytes_u_M_act = N_f_mix[ index_measurement_sel ][ index_mix ] * N_objects * sizeof( t_float_complex_gpu );

			// allocate device memory
			checkCudaErrors( cudaMalloc( (void **) &( d_u_M_float_complex[ index_measurement_sel ][ index_mix ] ), size_bytes_u_M_act ) );

			// initialize device memory
			if( mode == 1)
			{
				// set device memory to zero
				checkCudaErrors( cudaMemset( d_u_M_float_complex[ index_measurement_sel ][ index_mix ], 0, size_bytes_u_M_act ) );
			}
			else
			{
				// iterate objects
				if( DEBUG_MODE ) mexPrintf( "converting input_complex to float..." );
				for( int index_object = 0; index_object < N_objects; index_object++ )
				{

					// iterate frequencies
					for( int index_f = 0; index_f < N_f_mix[ index_measurement_sel ][ index_mix ]; index_f++ )
					{

						// compute destination index
						index_act = index_object * N_f_mix[ index_measurement_sel ][ index_mix ] + index_f;

						// compute source index
						index_src = N_observations_measurement_cs[ index_measurement_sel ] + N_f_mix_cs[ index_measurement_sel ][ index_mix ] + index_object * N_observations_sel + index_f;

						buffer_float_complex[ index_act ].x = (t_float_gpu) input_complex[ index_src ].real;
						buffer_float_complex[ index_act ].y = (t_float_gpu) input_complex[ index_src ].imag;

					} // for( int index_f = 0; index_f < N_f_mix[ index_measurement_sel ][ index_mix ]; index_f++ )

				} // for( int index_object = 0; index_object < N_objects; index_object++ )
				if( DEBUG_MODE ) mexPrintf( "done!\n" );

				// copy data
				checkCudaErrors( cudaMemcpy( d_u_M_float_complex[ index_measurement_sel ][ index_mix ], buffer_float_complex, size_bytes_u_M_act, cudaMemcpyHostToDevice ) );

			} // if( mode == 1)

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )

	} // for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )

	// device memory status
	if( DEBUG_MODE ) printMemInfo();

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 7.) forward or adjoint scattering
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// print status
	// time_start = tic;
	// str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
	// mexPrintf( "\t %s: quick adjoint scattering (GPU, Born approximation, single precision, kappa)...", str_date_time );

	// create cuBLAS handle
	checkCudaErrors( cublasCreate( &handle ) );

	// iterate selected sequential pulse-echo measurements
	for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )
	{

		// index of sequential pulse-echo measurement
		index_measurement = (int) indices_measurement_sel_double[ index_measurement_sel ] - 1;

		//--------------------------------------------------------------------------------------------------------------------------------------
		// a) extract incident acoustic pressure field and numbers of frequencies
		//--------------------------------------------------------------------------------------------------------------------------------------
		// extract incident acoustic pressure field
		p_incident_measurement = mxGetProperty( mxGetProperty( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "incident_waves" ), index_measurement, "p_incident" ), 0, "samples" ), 0, "values" );

		// ensure complex doubles (mxDOUBLE_CLASS)
		if( !( mxIsDouble( p_incident_measurement ) && mxIsComplex( p_incident_measurement ) ) ) mexErrMsgIdAndTxt( "combined_quick_gpu:NoComplexDoubles", "operator_born.incident_waves.p_incident must be complex doubles!" );

		// access complex doubles
		p_incident_measurement_complex = mxGetComplexDoubles( p_incident_measurement );

		//--------------------------------------------------------------------------------------------------------------------------------------
		// b) convert p_incident_measurement to float and copy to device (N_f_unique_measurement[ index_measurement_sel ] x N_points_occupied)
		//--------------------------------------------------------------------------------------------------------------------------------------
		// compute size
		size_bytes_p_incident_measurement = N_f_unique_measurement[ index_measurement_sel ] * N_points_occupied * sizeof( t_float_complex_gpu );
		if( DEBUG_MODE ) mexPrintf( "size_bytes_p_incident_measurement = %.2f MiB (%zu B)\n", ( (double) size_bytes_p_incident_measurement ) / BYTES_PER_MEBIBYTE, size_bytes_p_incident_measurement );

		// iterate matrix entries
		if( DEBUG_MODE ) mexPrintf( "converting p_incident_measurement to float..." );
		for( int index_point_occupied = 0; index_point_occupied < N_points_occupied; index_point_occupied++ )
		{
			for( int index_f = 0; index_f < N_f_unique_measurement[ index_measurement_sel ]; index_f++ )
			{
				index_act = index_point_occupied * N_f_unique_measurement[ index_measurement_sel ] + index_f;
				index_src = indices_grid_FOV_occupied[ index_point_occupied ] * N_f_unique_measurement[ index_measurement_sel ] + index_f;
				buffer_float_complex[ index_act ].x = (t_float_gpu) p_incident_measurement_complex[ index_src ].real;
				buffer_float_complex[ index_act ].y = (t_float_gpu) p_incident_measurement_complex[ index_src ].imag;
			}
		}
		if( DEBUG_MODE ) mexPrintf( "done!\n" );

		// copy data
		checkCudaErrors( cudaMemcpy2D( d_p_incident_measurement_float_complex, pitch_p_incident_measurement, buffer_float_complex, N_f_unique_measurement[ index_measurement_sel ] * sizeof( t_float_complex_gpu ), N_f_unique_measurement[ index_measurement_sel ] * sizeof( t_float_complex_gpu ), N_points_occupied, cudaMemcpyHostToDevice ) );

		// device memory status
		if( DEBUG_MODE ) printMemInfo();

		//--------------------------------------------------------------------------------------------------------------------------------------
		// e)
		//--------------------------------------------------------------------------------------------------------------------------------------
		// iterate mixed voltage signals
		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )
		{

			//----------------------------------------------------------------------------------------------------------------------------------
			// i.) parallelization settings
			//----------------------------------------------------------------------------------------------------------------------------------
			// number of blocks to process in parallel
			N_blocks_x = ceil( ( (double) N_points_occupied ) / N_THREADS_X );
			N_blocks_y = ceil( ( (double) N_f_mix[ index_measurement_sel ][ index_mix ] ) / N_THREADS_Y );
			dim3 numBlocks( N_blocks_x, N_blocks_y );

			//----------------------------------------------------------------------------------------------------------------------------------
			// ii.) main computations
			//----------------------------------------------------------------------------------------------------------------------------------
			// iterate active array elements
			for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement_sel ][ index_mix ]; index_active++ )
			{

				// index of active array element
				index_element = indices_active_mix[ index_measurement_sel ][ index_mix ][ index_active ];
				if( DEBUG_MODE ) mexPrintf( "index_element = %d\n", index_element );

				//------------------------------------------------------------------------------------------------------------------------------
				// compute entries of the observation matrix (N_f_mix[ index_measurement_sel ][ index_mix ] x N_points_occupied)
				//------------------------------------------------------------------------------------------------------------------------------
				compute_matrix_kernel<<<numBlocks, threadsPerBlock>>>(
					d_Phi_float_complex, N_f_mix[ index_measurement_sel ][ index_mix ], N_points_occupied,
					d_h_ref_float_complex, pitch_h_ref,
					d_indices_grid_FOV_shift[ index_element ],
					d_indices_f_mix_to_sequence[ index_measurement_sel ][ index_mix ],
					d_p_incident_measurement_float_complex, pitch_p_incident_measurement,
					d_indices_f_mix_to_measurement[ index_measurement_sel ][ index_mix ],
					d_prefactors_mix_float_complex[ index_measurement_sel ][ index_mix ][ index_active ]
				);

// TODO: canonical error checking
				// checkCudaErrors( cudaPeekAtLastError() );
				// checkCudaErrors( cudaDeviceSynchronize() );

				//------------------------------------------------------------------------------------------------------------------------------
				// compute matrix-matrix product (cuBLAS)
				//------------------------------------------------------------------------------------------------------------------------------
				// CUBLAS_OP_N: non-transpose operation / CUBLAS_OP_C: conjugate transpose operation
				if( mode == 1 )
				{
					// forward scattering
					checkCudaErrors(
						cublasCgemm( handle,
							CUBLAS_OP_N, CUBLAS_OP_N,
							N_f_mix[ index_measurement_sel ][ index_mix ], N_objects, N_points_occupied,
							&gemm_alpha, d_Phi_float_complex, N_f_mix[ index_measurement_sel ][ index_mix ], d_gamma_kappa_float_complex, N_points_occupied,
							&gemm_beta, d_u_M_float_complex[ index_measurement_sel ][ index_mix ], N_f_mix[ index_measurement_sel ][ index_mix ]
						)
					);
				}
				else
				{
					// adjoint scattering
					checkCudaErrors(
						cublasCgemm( handle,
							CUBLAS_OP_C, CUBLAS_OP_N,
							N_points_occupied, N_objects, N_f_mix[ index_measurement_sel ][ index_mix ],
							&gemm_alpha, d_Phi_float_complex, N_f_mix[ index_measurement_sel ][ index_mix ], d_u_M_float_complex[ index_measurement_sel ][ index_mix ], N_f_mix[ index_measurement_sel ][ index_mix ],
							&gemm_beta, d_gamma_kappa_float_complex, N_points_occupied
						)
					);
				} // if( mode == 1 )

				// clean-up device memory
				checkCudaErrors( cudaFree( d_prefactors_mix_float_complex[ index_measurement_sel ][ index_mix ][ index_active ] ) );

			} // for( int index_active = 0; index_active < N_elements_active_mix[ index_measurement_sel ][ index_mix ]; index_active++ )

			// clean-up memory
			mxFree( d_prefactors_mix_float_complex[ index_measurement_sel ][ index_mix ] );

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )

		// clean-up device memory
		mxFree( d_prefactors_mix_float_complex[ index_measurement_sel ] );

	} // for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )

	// destroy cuBLAS handle
	checkCudaErrors( cublasDestroy( handle ) );

	// clean-up host memory
	mxFree( d_prefactors_mix_float_complex );

	// clean-up device memory
	checkCudaErrors( cudaFree( d_p_incident_measurement_float_complex ) );
	checkCudaErrors( cudaFree( d_Phi_float_complex ) );
	checkCudaErrors( cudaFree( d_h_ref_float_complex ) );

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 8.) copy results to the host and convert to double
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if( mode == 1 )
	{
		// iterate selected sequential pulse-echo measurements
		for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )
		{

			// iterate mixed voltage signals
			for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )
			{

				// compute size
				size_bytes_u_M_act = N_f_mix[ index_measurement_sel ][ index_mix ] * N_objects * sizeof( t_float_complex_gpu );

				// copy data
				checkCudaErrors( cudaMemcpy( buffer_float_complex, d_u_M_float_complex[ index_measurement_sel ][ index_mix ], size_bytes_u_M_act, cudaMemcpyDeviceToHost ) );

				// iterate objects
				for( int index_object = 0; index_object < N_objects; index_object++ )
				{

					// iterate frequencies
					for( int index_f = 0; index_f < N_f_mix[ index_measurement_sel ][ index_mix ]; index_f++ )
					{

						// compute destination and source indices
						index_act = N_observations_measurement_cs[ index_measurement_sel ] + N_f_mix_cs[ index_measurement_sel ][ index_mix ] + index_object * N_observations_sel + index_f;
						index_src = index_object * N_f_mix[ index_measurement_sel ][ index_mix ] + index_f;

						output_complex[ index_act ].real = (mxDouble) buffer_float_complex[ index_src ].x;
						output_complex[ index_act ].imag = (mxDouble) buffer_float_complex[ index_src ].y;

					} // for( int index_f = 0; index_f < N_f_mix[ index_measurement_sel ][ index_mix ]; index_f++ )

				} // for( int index_object = 0; index_object < N_objects; index_object++ )

			} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )

		} // for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )
	}
	else
	{
		// copy data
		checkCudaErrors( cudaMemcpy( buffer_float_complex, d_gamma_kappa_float_complex, size_bytes_gamma_kappa, cudaMemcpyDeviceToHost ) );

		// iterate matrix entries
		if( DEBUG_MODE ) mexPrintf( "converting buffer_float_complex to mxDouble... " );
		for( int index_entry = 0; index_entry < N_points_occupied * N_objects; index_entry++ )
		{
			output_complex[ index_entry ].real = (mxDouble) buffer_float_complex[ index_entry ].x;
			output_complex[ index_entry ].imag = (mxDouble) buffer_float_complex[ index_entry ].y;
		}
		if( DEBUG_MODE ) mexPrintf( "done!\n" );

	} // if( mode == 1 )

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 9.) clean-up memory
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	checkCudaErrors( cudaFreeHost( buffer_float_complex ) );
	mxFree( indices_grid_FOV_occupied );

	// iterate selected sequential pulse-echo measurements
	for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )
	{

		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )
		{

			checkCudaErrors( cudaFree( d_u_M_float_complex[ index_measurement_sel ][ index_mix ] ) );

		}

		mxFree( d_u_M_float_complex[ index_measurement_sel ] );

	}

	mxFree( d_u_M_float_complex );
	checkCudaErrors( cudaFree( d_gamma_kappa_float_complex ) );

	for( int index_element = 0; index_element < N_elements; index_element++ )
	{

		checkCudaErrors( cudaFree( d_indices_grid_FOV_shift[ index_element ] ) );
		

	} // for( int index_element = 0; index_element < N_elements; index_element++ )

	mxFree( d_indices_grid_FOV_shift );
	

	for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )
	{

		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )
		{

			checkCudaErrors( cudaFree( d_indices_f_mix_to_sequence[ index_measurement_sel ][ index_mix ] ) );
			checkCudaErrors( cudaFree( d_indices_f_mix_to_measurement[ index_measurement_sel ][ index_mix ] ) );

			mxFree( indices_active_mix[ index_measurement_sel ][ index_mix ] );
			mxFree( indices_f_mix_to_sequence[ index_measurement_sel ][ index_mix ] );
			mxFree( indices_f_mix_to_measurement[ index_measurement_sel ][ index_mix ] );

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement_sel ]; index_mix++ )

		mxFree( prefactors_mix_complex[ index_measurement_sel ] );

		mxFree( d_indices_f_mix_to_sequence[ index_measurement_sel ] );
		mxFree( d_indices_f_mix_to_measurement[ index_measurement_sel ] );

		mxFree( indices_active_mix[ index_measurement_sel ] );
		mxFree( indices_f_mix_to_sequence[ index_measurement_sel ] );
		mxFree( indices_f_mix_to_measurement[ index_measurement_sel ] );

		mxFree( N_f_mix[ index_measurement_sel ] );
		mxFree( N_f_mix_cs[ index_measurement_sel ] );
		mxFree( N_elements_active_mix[ index_measurement_sel ] );

	} // for( int index_measurement_sel = 0; index_measurement_sel < N_measurements_sel; index_measurement_sel++ )

	mxFree( prefactors_mix_complex );
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
__global__ void compute_matrix_kernel( t_float_complex_gpu* d_Phi_float_complex, int N_f_mix, int N_points_occupied, t_float_complex_gpu* d_h_ref_float_complex, size_t pitch_h_ref, int* d_indices_grid_FOV_shift, int* d_indices_f_mix_to_sequence, t_float_complex_gpu* d_p_incident_measurement_float_complex, size_t pitch_p_incident_measurement, int* d_indices_f_mix_to_measurement, t_float_complex_gpu* d_prefactors_mix_float_complex )
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
												d_prefactors_mix_float_complex[ index_f ]
											);

	} // if( index_f < N_f_mix && index_point_occupied < N_points_occupied )

} // __global__ void compute_matrix_kernel( t_float_complex_gpu* d_Phi_float_complex, int N_f_mix, int N_points_occupied, t_float_complex_gpu* d_h_ref_float_complex, size_t pitch_h_ref, int* d_indices_grid_FOV_shift, int* d_indices_f_mix_to_sequence, t_float_complex_gpu* d_p_incident_measurement_float_complex, size_t pitch_p_incident_measurement, int* d_indices_f_mix_to_measurement, t_float_complex_gpu* d_prefactors_mix_float_complex )

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