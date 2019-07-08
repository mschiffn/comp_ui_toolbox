//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// adjoint_quick_gpu.cu
//
// implementation of
// the adjoint scattering operator (N_points x N_observations) using
// the reference spatial transfer function
//
// 0. gamma_hat						adjoint relative spatial fluctuations in compressibility (N_points x N_objects)
// = adjoint_quick_gpu
// (
// 0. operator_born,				object of class scattering.operator_born (scalar)
// 1. u_M,							mixed voltage signals (N_observations x N_objects)
// 2. index_device,					device index to be used (scalar, default: 0)
// 3. spatial_aliasing				anti-aliasing (default: 0)
// )
//
// author: Martin F. Schiffner
// date: 2019-06-29
// modified: 2019-07-08
// All rights reserved!
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// The Thrust template library can do this trivially using thrust::transform, for example:
// thrust::multiplies<thrust::complex<float> > op;
// thrust::transform(thrust::device, x, x + n, y, z, op);
// would iterate over each pair of inputs from the device pointers x and y and calculate z[i] = x[i] * y[i]
// (there is probably a couple of casts you need to make to compile that, but you get the idea). But that effectively requires compilation of CUDA code within your project, and apparently you don't want that.

// TODO: mwSize vs size_t in mxMalloc
// TODO: make h_ref persistent make

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

	// parallelization
	int N_blocks_x = 0, N_blocks_y = 0;					// numbers of blocks along 1st and 2nd dimension

	// dimensions of output vector
	mwSize dimensions_output[ 2 ];

	// misc variables
	mxArray* discretization = NULL;
	mxArray* discretization_spectral = NULL;
	mxArray* temp = NULL;

	// device status
	size_t size_bytes_gpu_available = 0;
	size_t size_bytes_gpu_total = 0;
	int index_device = 0;
	int deviceCount = 0;

	// reference spatial transfer function
	mxArray* h_ref = NULL;
	mxComplexDouble* h_ref_complex = NULL;

	size_t size_bytes_h_ref = 0, pitch_h_ref = 0;
	t_float_complex_gpu* h_ref_float_complex = NULL;

	// indices of shifted grid points
	mxArray* indices_grid_FOV_shift = NULL;
	mxDouble* indices_grid_FOV_shift_double = NULL;

	size_t size_bytes_indices_grid_FOV_shift = 0, pitch_indices_grid_FOV_shift = 0;
	int* indices_grid_FOV_shift_int = 0;

	// incident acoustic pressure field
	mxArray* p_incident_measurement = NULL;
	mxComplexDouble * p_incident_measurement_complex = NULL;

	size_t size_bytes_p_incident_measurement = 0, pitch_p_incident_measurement = 0;
	t_float_complex_gpu *p_incident_measurement_float_complex = NULL;

	// prefactors for current mix
	mxArray* prefactors_mix = NULL;
	mxComplexDouble * prefactors_mix_complex = NULL;

	size_t size_bytes_prefactors_mix = 0, pitch_prefactors_mix = 0;
	t_float_complex_gpu *prefactors_mix_float_complex = NULL;

	//
	mxArray* rx_measurement = NULL;
	mxArray* prefactors_measurement = NULL;

	double* indices_active_mix = NULL;
	int index_element = 0;
	int index_act = 0, index_src = 0;

	// mixed voltage signals
	mxComplexDouble* u_M_complex = NULL;

	size_t size_bytes_u_M = 0, size_bytes_u_M_act = 0;
	t_float_complex_gpu*** u_M_float_complex = NULL;

	// adjoint relative spatial fluctuations
	mxComplexDouble* gamma_hat_complex = NULL;

	size_t size_bytes_gamma_hat = 0;
	t_float_complex_gpu* gamma_hat_float_complex = NULL;

	// device variables
	t_float_complex_gpu* d_h_ref_float_complex = NULL;
	int* d_indices_grid_FOV_shift_int = NULL;
	t_float_complex_gpu* d_p_incident_measurement_float_complex = NULL;
	t_float_complex_gpu* d_prefactors_mix_float_complex = NULL;

	size_t size_bytes_indices_f_act = 0;
	int*** d_indices_f_mix_to_measurement = NULL;
	int*** d_indices_f_mix_to_sequence = NULL;

	size_t size_bytes_Phi_max = 0, size_bytes_Phi = 0;
	t_float_complex_gpu* d_Phi_float_complex = NULL;

	t_float_complex_gpu*** d_u_M_float_complex = NULL;

	t_float_complex_gpu* d_gamma_hat_float_complex = NULL;

	dim3 threadsPerBlock( N_THREADS_X, N_THREADS_Y );

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 2.) check arguments
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// check for proper number of arguments
	if ( nrhs != 3 || nlhs != 1 )
	{
		mexErrMsgIdAndTxt( "adjoint_quick_gpu:NumberArguments", "adjoint_quick_gpu requires 3 inputs and 1 output." );
	}

	// ensure class scattering.operator_born (scalar)
	if( mxIsClass( prhs[ 0 ], "scattering.operator_born" ) && mxGetNumberOfElements( prhs[ 0 ] ) == 1 )
	{

		discretization = mxGetProperty( prhs[ 0 ], 0, "discretization" );
		discretization_spectral = mxGetProperty( discretization, 0, "spectral" );

		// number of grid points
		N_points = (int) mxGetScalar( mxGetProperty( mxGetProperty( mxGetProperty( discretization, 0, "spatial" ), 0, "grid_FOV" ), 0, "N_points" ) );
		if( DEBUG_MODE ) mexPrintf( "N_points = %d\n", N_points );

		// number of sequential pulse-echo measurements
		N_measurements = mxGetNumberOfElements( discretization_spectral );
		if( DEBUG_MODE ) mexPrintf( "N_measurements = %d\n", N_measurements );

		// numbers of mixed voltage signals and frequencies per mix
		N_mix_measurement = (int*) mxMalloc( N_measurements * sizeof( int ) );
		N_f_unique_measurement = (int*) mxMalloc( N_measurements * sizeof( int ) );
		N_f_mix = (int**) mxMalloc( N_measurements * sizeof( int* ) );
		N_f_mix_cs = (int**) mxMalloc( N_measurements * sizeof( int* ) );
		N_elements_active_mix = (int**) mxMalloc( N_measurements * sizeof( int* ) );
		N_observations_measurement = (int*) mxMalloc( N_measurements * sizeof( int ) );
		N_observations_measurement_cs = (int*) mxCalloc( N_measurements, sizeof( int ) );
		indices_f_mix_to_measurement = (int***) mxMalloc( N_measurements * sizeof( int** ) );
		indices_f_mix_to_sequence = (int***) mxMalloc( N_measurements * sizeof( int** ) );

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

			// map unique frequencies of pulse-echo measurement to global unique frequencies
			indices_f_measurement_to_sequence_double = (double*) mxGetData( mxGetCell( mxGetProperty( discretization, 0, "indices_f_to_unique" ), (mwIndex) index_measurement ) );

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
				N_elements_active_mix[ index_measurement ][ index_mix ] = mxGetNumberOfElements( mxGetProperty( mxGetProperty( discretization_spectral, index_measurement, "rx" ), index_mix, "indices_active" ) );
				if( DEBUG_MODE ) mexPrintf( "N_elements_active_mix[%d][%d] = %d\n", index_measurement, index_mix, N_elements_active_mix[ index_measurement ][ index_mix ] );

				// map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
				indices_f_mix_to_measurement_double = (double*) mxGetData( mxGetCell( mxGetProperty( discretization_spectral, index_measurement, "indices_f_to_unique" ), (mwIndex) index_mix ) );

				// allocate memory
				indices_f_mix_to_measurement[ index_measurement ][ index_mix ] = (int*) mxMalloc( N_f_mix[ index_measurement ][ index_mix ] * sizeof( int ) );
				indices_f_mix_to_sequence[ index_measurement ][ index_mix ] = (int*) mxMalloc( N_f_mix[ index_measurement ][ index_mix ] * sizeof( int ) );

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

	}
	else
	{
		mexErrMsgIdAndTxt( "adjoint_quick_gpu:NoOperatorBorn", "operator_born must be a single scattering.operator_born!" );
	} // if ( mxIsClass( prhs[ 0 ], "scattering.operator_born" ) && mxGetNumberOfElements( prhs[ 0 ] ) == 1 )

	// ensure numeric matrix
	if ( mxIsNumeric( prhs[ 1 ] ) )
	{
		u_M_complex = mxGetComplexDoubles( prhs[ 1 ] );

		// number of observations
		N_observations = mxGetM( prhs[ 1 ] );
		if( DEBUG_MODE ) mexPrintf( "N_observations = %d\n", N_observations );

		// number of objects
		N_objects = mxGetN( prhs[ 1 ] );
		if( DEBUG_MODE ) mexPrintf( "N_objects = %d\n", N_objects );
	}
	else
	{
		mexErrMsgIdAndTxt( "adjoint_quick_gpu:NoNumericMatrix", "u_M must be a numeric matrix!" );
	} // if ( mxIsNumeric( prhs[ 1 ] ) )

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 3.) check number of GPUs and their capabilities, print GPU information
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// TODO: outsource to function
	cudaDeviceProp deviceProp;
	int driverVersion = 0;       // driver version
	int runtimeVersion = 0;      // runtime version
	int capability_double = 0;   // capability for double precision?

	// check number of devices supporting CUDA
	if ( cudaGetDeviceCount( &deviceCount ) != cudaSuccess )
	{
		// print error message, exit program
		mexErrMsgIdAndTxt( "adjoint_quick_gpu:DeviceCountFailed", "Could not allocate memory for d_h_ref_float_complex on device!" );
		//printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
	}
	if( DEBUG_MODE ) mexPrintf( "number of CUDA devices: %d\n", deviceCount );

	// if no CUDA-enabled GPU was detected
	if ( deviceCount < 1 ) mexErrMsgIdAndTxt( "adjoint_quick_gpu:NoCUDADevices", "Could not find CUDA capable GPU!" );

	// if selected GPU does not exist
	if( DEBUG_MODE ) mexPrintf( "using device: %d\n", index_device );
	if ( index_device >= deviceCount || index_device < 0 ) mexErrMsgIdAndTxt( "adjoint_quick_gpu:InvalidCUDADevice", "Invalid device selected!" );
	// assertion: deviceCount > 0 && 0 <= index_device < deviceCount

	// check properties of chosen GPU
	cudaGetDeviceProperties( &deviceProp, index_device );
// 	if( DEBUG_MODE ) print_device_info( index_device, deviceProp );

	if(deviceProp.major >= 1 && deviceProp.minor >= 3)
	{
		capability_double = 1;
	}
	if( DEBUG_MODE ) mexPrintf( "\tcapability for double precision: %d\n", capability_double );

	// set device to operate on
	cudaSetDevice( index_device );

	// get driver version
	cudaDriverGetVersion( &driverVersion );
	if(DEBUG_MODE) mexPrintf("\ndriver version: %d\n", driverVersion);
    
	// get runtime version
	cudaRuntimeGetVersion( &runtimeVersion );
	if(DEBUG_MODE) mexPrintf("runtime version: %d\n", runtimeVersion);

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 4.)
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	//---------------------------------------------------------------------
	// a) extract reference spatial transfer function
	//---------------------------------------------------------------------
	// extract reference spatial transfer function
// TODO: anti-aliasing?
	mxArray *mxi;
	temp = mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "options" ), 0, "spatial_aliasing" );
	mexCallMATLAB( 1, &mxi, 1, &temp, "int32" );

	if(DEBUG_MODE) mexPrintf( "mxi = %d\n", *( (int*) mxGetData( mxi ) ) );

	if( mxIsClass( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "options" ), 0, "spatial_aliasing" ), "scattering.options_aliasing.exclude" ) )
	{
		h_ref = mxGetProperty( mxGetProperty( mxGetProperty( discretization, 0, "h_ref_aa" ), 0, "samples" ), 0, "values" );
		if(DEBUG_MODE) mexPrintf( "options.spatial_aliasing = %s\n", mxGetClassName( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "options" ), 0, "spatial_aliasing" ) ) );
	}
	else
	{
		h_ref = mxGetProperty( mxGetProperty( mxGetProperty( discretization, 0, "h_ref" ), 0, "samples" ), 0, "values" );
		if(DEBUG_MODE) mexPrintf( "options.spatial_aliasing = %s\n", mxGetClassName( mxGetProperty( mxGetProperty( prhs[ 0 ], 0, "options" ), 0, "spatial_aliasing" ) ) );
	}
	h_ref_complex = mxGetComplexDoubles( h_ref );

	// number of unique frequencies
	N_f_unique = mxGetM( h_ref );
	if( DEBUG_MODE ) mexPrintf( "N_f_unique = %d\n", N_f_unique );

	//---------------------------------------------------------------------
	// b) convert h_ref_complex to float (if necessary)
	//---------------------------------------------------------------------
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

	//---------------------------------------------------------------------
	// c) extract indices of shifted grid points
	//---------------------------------------------------------------------
	// extract indices of shifted grid points
	indices_grid_FOV_shift = mxGetProperty( discretization, 0, "indices_grid_FOV_shift" );
	indices_grid_FOV_shift_double = mxGetDoubles( indices_grid_FOV_shift );

	// number of array elements
	N_elements = mxGetN( indices_grid_FOV_shift );
	if( DEBUG_MODE ) mexPrintf( "N_elements = %d\n", N_elements );

	//---------------------------------------------------------------------
	// d) convert indices_grid_FOV_shift_double to int
	//---------------------------------------------------------------------
	// compute size
	size_bytes_indices_grid_FOV_shift = N_points * N_elements * sizeof( int );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_indices_grid_FOV_shift = %.2f MiB (%zu B)\n", ( ( double ) size_bytes_indices_grid_FOV_shift ) / BYTES_PER_MEBIBYTE, size_bytes_indices_grid_FOV_shift );

	// allocate memory
	indices_grid_FOV_shift_int = (int*) mxMalloc( size_bytes_indices_grid_FOV_shift );

	// iterate elements
	if( DEBUG_MODE ) mexPrintf( "converting indices_grid_FOV_shift_double to int..." );
	for ( int index_element = 0; index_element < N_elements; index_element++ )
	{
		for ( int index_point = 0; index_point < N_points; index_point++ )
		{
			index_act = index_element * N_points + index_point;
			indices_grid_FOV_shift_int[ index_act ] = (int) indices_grid_FOV_shift_double[ index_act ] - 1;
		}
	}
	if( DEBUG_MODE ) mexPrintf( "done!\n" );

	//---------------------------------------------------------------------
	// e)
	//---------------------------------------------------------------------

	//---------------------------------------------------------------------
	// f) convert u_M_complex to float (N_observations x N_objects)
	//---------------------------------------------------------------------
	// compute size
	size_bytes_u_M = N_observations * N_objects * sizeof( t_float_complex_gpu );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_u_M = %.2f MiB (%zu B)\n", ( ( double ) size_bytes_u_M ) / BYTES_PER_MEBIBYTE, size_bytes_u_M );

	// allocate memory
	u_M_float_complex = (t_float_complex_gpu***) mxMalloc( N_measurements * sizeof( t_float_complex_gpu** ) );

	// iterate sequential pulse-echo measurements
	if( DEBUG_MODE ) mexPrintf( "converting u_M_complex to float..." );
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

					u_M_float_complex[ index_measurement ][ index_mix ][ index_act ].x = (t_float_gpu) u_M_complex[ index_src ].real;
					u_M_float_complex[ index_measurement ][ index_mix ][ index_act ].y = (t_float_gpu) u_M_complex[ index_src ].imag;

				} // for( int index_f = 0; index_f < N_f_mix[ index_measurement ][ index_mix ]; index_f++ )

			} // for( int index_object = 0; index_object < N_objects; index_object++ )

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

	} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
	if( DEBUG_MODE ) mexPrintf( "done!\n" );

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 5.) copy data to the device
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	//---------------------------------------------------------------------
	// a) copy h_ref_float_complex to the device
	//---------------------------------------------------------------------
	// allocate memory
	checkCudaErrors( cudaMallocPitch( (void **) &d_h_ref_float_complex, &pitch_h_ref, N_f_unique * sizeof( t_float_complex_gpu ), N_points ) );

	// copy data
	checkCudaErrors( cudaMemcpy2D( d_h_ref_float_complex, pitch_h_ref, h_ref_float_complex, N_f_unique * sizeof( t_float_complex_gpu ), N_f_unique * sizeof( t_float_complex_gpu ), N_points, cudaMemcpyHostToDevice ) );

	// clean-up host memory
	mxFree( h_ref_float_complex );

	// memory status
	if( DEBUG_MODE ) printMemInfo();

	//---------------------------------------------------------------------
	// b) copy indices_grid_FOV_shift_int to the device
	//---------------------------------------------------------------------
	// allocate memory
	checkCudaErrors( cudaMallocPitch( (void **) &d_indices_grid_FOV_shift_int, &pitch_indices_grid_FOV_shift, N_points * sizeof( int ), N_elements ) );

	// copy data
	checkCudaErrors( cudaMemcpy2D( d_indices_grid_FOV_shift_int, pitch_indices_grid_FOV_shift, indices_grid_FOV_shift_int, N_points * sizeof( int ), N_points * sizeof( int ), N_elements, cudaMemcpyHostToDevice ) );

	// clean-up host memory
	mxFree( indices_grid_FOV_shift_int );

	// memory status
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

	// memory status
	if( DEBUG_MODE ) printMemInfo();

	//---------------------------------------------------------------------
	// d) allocate memory for d_p_incident_measurement_float_complex (use maximum size: N_f_unique_measurement_max x N_points)
	//---------------------------------------------------------------------
	// compute size
	// allocate memory
	// memory status

	//---------------------------------------------------------------------
	// e) copy u_M_float_complex to the device
	//---------------------------------------------------------------------
	// allocate memory
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

			// allocate device memory
			checkCudaErrors( cudaMalloc( (void **) &( d_u_M_float_complex[ index_measurement ][ index_mix ] ), size_bytes_u_M_act ) );

			// copy data
			checkCudaErrors( cudaMemcpy( d_u_M_float_complex[ index_measurement ][ index_mix ], u_M_float_complex[ index_measurement ][ index_mix ], size_bytes_u_M_act, cudaMemcpyHostToDevice ) );

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

	} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )

	// memory status
	if( DEBUG_MODE ) printMemInfo();

	//---------------------------------------------------------------------
	// f) allocate memory for d_Phi_float_complex (use maximum size: N_f_mix_max x N_points)
	//---------------------------------------------------------------------
	// compute size
	size_bytes_Phi_max = N_f_mix_max * N_points * sizeof( t_float_complex_gpu );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_Phi_max = %.2f MiB (%zu B)\n", ( (double) size_bytes_Phi_max ) / BYTES_PER_MEBIBYTE, size_bytes_Phi_max );

	// allocate memory
	checkCudaErrors( cudaMalloc( (void **) &d_Phi_float_complex, size_bytes_Phi_max ) );

	// memory status
	if( DEBUG_MODE ) printMemInfo();

	//---------------------------------------------------------------------
	// g) allocate and initialize memory for d_gamma_hat_float_complex
	//---------------------------------------------------------------------
	// compute size
	size_bytes_gamma_hat = N_points * N_objects * sizeof( t_float_complex_gpu );
	if( DEBUG_MODE ) mexPrintf( "size_bytes_gamma_hat = %.2f MiB (%zu B)\n", ( ( double ) size_bytes_gamma_hat ) / BYTES_PER_MEBIBYTE, size_bytes_gamma_hat );

	// allocate and initialize memory
	checkCudaErrors( cudaMalloc( (void **) &d_gamma_hat_float_complex, size_bytes_gamma_hat ) );
	checkCudaErrors( cudaMemset( d_gamma_hat_float_complex, 0, size_bytes_gamma_hat ) );

	// memory status
	if( DEBUG_MODE ) printMemInfo();

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 6.) compute adjoint fluctuations
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// cuBLAS settings
	cublasHandle_t handle;
	const t_float_complex_gpu gemm_alpha = make_cuFloatComplex( 1.0f, 0.0f );
	const t_float_complex_gpu gemm_beta = make_cuFloatComplex( 1.0f, 0.0f );

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
		// b) convert p_incident_measurement to float
		//-----------------------------------------------------------------
		// compute size
		size_bytes_p_incident_measurement = N_f_unique_measurement[ index_measurement ] * N_points * sizeof( t_float_complex_gpu );
		if( DEBUG_MODE ) mexPrintf( "size_bytes_p_incident_measurement = %.2f MiB (%zu B)\n", ( (double) size_bytes_p_incident_measurement ) / BYTES_PER_MEBIBYTE, size_bytes_p_incident_measurement );

		// allocate memory
		p_incident_measurement_float_complex = (t_float_complex_gpu *) mxMalloc( size_bytes_p_incident_measurement );

		// iterate elements
		if( DEBUG_MODE ) mexPrintf( "converting p_incident_measurement to float..." );
		for ( int index_point = 0; index_point < N_points; index_point++ )
		{
			for ( int index_f = 0; index_f < N_f_unique_measurement[ index_measurement ]; index_f++ )
			{
				index_act = index_point * N_f_unique_measurement[ index_measurement ] + index_f;
				p_incident_measurement_float_complex[ index_act ].x = (t_float_gpu) p_incident_measurement_complex[ index_act ].real;
				p_incident_measurement_float_complex[ index_act ].y = (t_float_gpu) p_incident_measurement_complex[ index_act ].imag;
			}
		}
		if( DEBUG_MODE ) mexPrintf( "done!\n" );

		//-----------------------------------------------------------------
		// c) copy p_incident_measurement_float_complex to the device
		//-----------------------------------------------------------------
		// allocate memory
		checkCudaErrors( cudaMallocPitch( (void **) &d_p_incident_measurement_float_complex, &pitch_p_incident_measurement, N_f_unique_measurement[ index_measurement ] * sizeof( t_float_complex_gpu ), N_points ) );

		// copy data
		checkCudaErrors( cudaMemcpy2D( d_p_incident_measurement_float_complex, pitch_p_incident_measurement, p_incident_measurement_float_complex, N_f_unique_measurement[ index_measurement ] * sizeof( t_float_complex_gpu ), N_f_unique_measurement[ index_measurement ] * sizeof( t_float_complex_gpu ), N_points, cudaMemcpyHostToDevice ) );

		// clean-up host memory
		mxFree( p_incident_measurement_float_complex );

		// memory status
		if( DEBUG_MODE ) printMemInfo();

		//-----------------------------------------------------------------
		// d)
		//-----------------------------------------------------------------
		// extract rx settings
		rx_measurement = mxGetProperty( discretization_spectral, index_measurement, "rx" );

		// extract prefactors for all mixes
		prefactors_measurement = mxGetCell( mxGetProperty( discretization, 0, "prefactors" ), (mwIndex) index_measurement );

		//-----------------------------------------------------------------
		// e)
		//-----------------------------------------------------------------
		// iterate mixed voltage signals
		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
		{

			//-------------------------------------------------------------
			// i.) extract indices of active array elements
			//-------------------------------------------------------------
			// extract indices_active_mix
			temp = mxGetProperty( rx_measurement, index_mix, "indices_active" );
			indices_active_mix = (double *) mxGetData( temp );

			//-------------------------------------------------------------
			// ii.) extract prefactors for current mix
			//-------------------------------------------------------------
			// extract prefactors_mix
			prefactors_mix = mxGetProperty( mxGetProperty( prefactors_measurement, index_mix, "samples" ), 0, "values" );
			prefactors_mix_complex = mxGetComplexDoubles( prefactors_mix );

			//-------------------------------------------------------------
			// iii.) convert prefactors_mix to float
			//-------------------------------------------------------------
			// compute size
			size_bytes_prefactors_mix = N_f_mix[ index_measurement ][ index_mix ] * N_elements_active_mix[ index_measurement ][ index_mix ] * sizeof( t_float_complex_gpu );
			if( DEBUG_MODE ) mexPrintf( "size_bytes_prefactors_mix = %.2f kiB (%zu B)\n", ( (double) size_bytes_prefactors_mix ) / BYTES_PER_KIBIBYTE, size_bytes_prefactors_mix );

			// allocate memory
			prefactors_mix_float_complex = (t_float_complex_gpu *) mxMalloc( size_bytes_prefactors_mix );

			// iterate elements
			if( DEBUG_MODE ) mexPrintf( "converting prefactors_mix to float..." );
			for ( int index_active = 0; index_active < N_elements_active_mix[ index_measurement ][ index_mix ]; index_active++ )
			{
				for ( int index_f = 0; index_f < N_f_mix[ index_measurement ][ index_mix ]; index_f++ )
				{
					index_act = index_active * N_f_mix[ index_measurement ][ index_mix ] + index_f;
					prefactors_mix_float_complex[ index_act ].x = (t_float_gpu) prefactors_mix_complex[ index_act ].real;
					prefactors_mix_float_complex[ index_act ].y = (t_float_gpu) prefactors_mix_complex[ index_act ].imag;
				}
			}
			if( DEBUG_MODE ) mexPrintf( "done!\n" );

			//-------------------------------------------------------------
			// iv.) copy prefactors_mix_float_complex to the device
			//-------------------------------------------------------------
			// allocate memory
			checkCudaErrors( cudaMallocPitch( (void **) &d_prefactors_mix_float_complex, &pitch_prefactors_mix, N_f_mix[ index_measurement ][ index_mix ] * sizeof( t_float_complex_gpu ), N_elements_active_mix[ index_measurement ][ index_mix ] ) );

			// copy data
			checkCudaErrors( cudaMemcpy2D( d_prefactors_mix_float_complex, pitch_prefactors_mix, prefactors_mix_float_complex, N_f_mix[ index_measurement ][ index_mix ] * sizeof( t_float_complex_gpu ), N_f_mix[ index_measurement ][ index_mix ] * sizeof( t_float_complex_gpu ), N_elements_active_mix[ index_measurement ][ index_mix ], cudaMemcpyHostToDevice ) );

			// clean-up host memory
			mxFree( prefactors_mix_float_complex );

			// memory status
			if( DEBUG_MODE ) printMemInfo();

			//-------------------------------------------------------------
			// v.)
			//-------------------------------------------------------------
			// number of blocks to process in parallel
			N_blocks_x = ceil( ( (double) N_points ) / N_THREADS_X );
			N_blocks_y = ceil( ( (double) N_f_mix[ index_measurement ][ index_mix ] ) / N_THREADS_Y );
			dim3 numBlocks( N_blocks_x, N_blocks_y );

			// iterate active array elements
			for ( int index_active = 0; index_active < N_elements_active_mix[ index_measurement ][ index_mix ]; index_active++ )
			{

				// index of active array element
				index_element = ( int ) indices_active_mix[ index_active ] - 1;
				if( DEBUG_MODE ) mexPrintf( "index_element = %d\n", index_element );

				// compute entries of the observation matrix (N_f_mix[ index_measurement ][ index_mix ] x N_points)
				compute_matrix_kernel<<<numBlocks, threadsPerBlock>>>(
					d_Phi_float_complex, N_f_mix[ index_measurement ][ index_mix ], N_points,
					d_h_ref_float_complex, pitch_h_ref,
					d_indices_grid_FOV_shift_int, pitch_indices_grid_FOV_shift, index_element,
					d_indices_f_mix_to_sequence[ index_measurement ][ index_mix ],
					d_p_incident_measurement_float_complex, pitch_p_incident_measurement,
					d_indices_f_mix_to_measurement[ index_measurement ][ index_mix ],
					d_prefactors_mix_float_complex, pitch_prefactors_mix,
					index_active
				);

// TODO: canonical error checking
				// checkCudaErrors( cudaPeekAtLastError() );
				// checkCudaErrors( cudaDeviceSynchronize() );

				// compute matrix-matrix product (cuBLAS)
				// CUBLAS_OP_N: non-transpose operation / CUBLAS_OP_T: transpose operation / CUBLAS_OP_C: conjugate transpose operation
				checkCudaErrors(
					cublasCgemm( handle,
						CUBLAS_OP_C, CUBLAS_OP_N,
						N_points, N_objects, N_f_mix[ index_measurement ][ index_mix ],
						&gemm_alpha, d_Phi_float_complex, N_f_mix[ index_measurement ][ index_mix ], d_u_M_float_complex[ index_measurement ][ index_mix ], N_f_mix[ index_measurement ][ index_mix ],
						&gemm_beta, d_gamma_hat_float_complex, N_points
					)
				);

			} // for ( index_active = 0; index_active < N_elements_active_mix[ index_measurement ][ index_mix ]; index_active++ )

			// clean-up device memory
			checkCudaErrors( cudaFree( d_prefactors_mix_float_complex ) );

		} // for ( index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

		// clean-up device memory
		checkCudaErrors( cudaFree( d_p_incident_measurement_float_complex ) );

	} // for index_measurement = 1:numel( operator_born.discretization.spectral )

	// destroy cuBLAS handle
	checkCudaErrors( cublasDestroy( handle ) );

	// clean-up device memory
	checkCudaErrors( cudaFree( d_Phi_float_complex ) );
	checkCudaErrors( cudaFree( d_indices_grid_FOV_shift_int ) );
	checkCudaErrors( cudaFree( d_h_ref_float_complex ) );

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 7.) copy results to the host
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	//---------------------------------------------------------------------
	// a) copy d_gamma_hat_float_complex to the host
	//---------------------------------------------------------------------
	// allocate memory
	gamma_hat_float_complex = (t_float_complex_gpu *) mxMalloc( size_bytes_gamma_hat );

	// copy data
	checkCudaErrors( cudaMemcpy( gamma_hat_float_complex, d_gamma_hat_float_complex, size_bytes_gamma_hat, cudaMemcpyDeviceToHost ) );

	// clean-up device memory
	checkCudaErrors( cudaFree( d_gamma_hat_float_complex ) );

	// device memory status
	if( DEBUG_MODE )
	{
		cudaMemGetInfo( &size_bytes_gpu_available, &size_bytes_gpu_total );
		mexPrintf( "memory available on GPU: %.2f MiB (%zu B)\n", ( ( double ) size_bytes_gpu_available / BYTES_PER_MEBIBYTE ), size_bytes_gpu_available );
	}

	//---------------------------------------------------------------------
	// b) convert gamma_hat_float_complex to double
	//---------------------------------------------------------------------
	// allocate workspace memory
	dimensions_output[0] = N_points;
	dimensions_output[1] = N_objects;

	plhs[ 0 ] = mxCreateNumericArray( 2, dimensions_output, mxDOUBLE_CLASS, mxCOMPLEX );
	gamma_hat_complex = ( mxComplexDouble* ) mxGetData( plhs[0] );

	// iterate elements
	if( DEBUG_MODE ) mexPrintf( "converting gamma_hat_float_complex to double..." );
	for ( int index_object = 0; index_object < N_objects; index_object++ )
	{
		for ( int index_point = 0; index_point < N_points; index_point++ )
		{
			index_act = index_object * N_points + index_point;
			gamma_hat_complex[ index_act ].real = (mxDouble) gamma_hat_float_complex[ index_act ].x;
			gamma_hat_complex[ index_act ].imag = (mxDouble) gamma_hat_float_complex[ index_act ].y;
		}
	}
	if( DEBUG_MODE ) mexPrintf( "done!\n" );

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 8.) clean-up memory
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )
	{

		for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )
		{
			checkCudaErrors( cudaFree( d_u_M_float_complex[ index_measurement ][ index_mix ] ) );
			mxFree( u_M_float_complex[ index_measurement ][ index_mix ] );

			checkCudaErrors( cudaFree( d_indices_f_mix_to_sequence[ index_measurement ][ index_mix ] ) );
			checkCudaErrors( cudaFree( d_indices_f_mix_to_measurement[ index_measurement ][ index_mix ] ) );

			mxFree( indices_f_mix_to_sequence[ index_measurement ][ index_mix ] );
			mxFree( indices_f_mix_to_measurement[ index_measurement ][ index_mix ] );

		} // for( int index_mix = 0; index_mix < N_mix_measurement[ index_measurement ]; index_mix++ )

		mxFree( d_u_M_float_complex[ index_measurement ] );
		mxFree( u_M_float_complex[ index_measurement ] );

		mxFree( d_indices_f_mix_to_sequence[ index_measurement ] );
		mxFree( d_indices_f_mix_to_measurement[ index_measurement ] );
		mxFree( indices_f_mix_to_sequence[ index_measurement ] );
		mxFree( indices_f_mix_to_measurement[ index_measurement ] );

		mxFree( N_f_mix[ index_measurement ] );
		mxFree( N_f_mix_cs[ index_measurement ] );
		mxFree( N_elements_active_mix[ index_measurement ] );

	} // for( int index_measurement = 0; index_measurement < N_measurements; index_measurement++ )

	mxFree( d_u_M_float_complex );
	mxFree( u_M_float_complex );

	mxFree( d_indices_f_mix_to_sequence );
	mxFree( d_indices_f_mix_to_measurement );
	mxFree( indices_f_mix_to_sequence );
	mxFree( indices_f_mix_to_measurement );

	mxFree( N_f_mix );
	mxFree( N_f_mix_cs );
	mxFree( N_elements_active_mix );

	mxFree( N_mix_measurement );
	mxFree( N_f_unique_measurement );
	mxFree( N_observations_measurement );
	mxFree( N_observations_measurement_cs );

	mxFree( gamma_hat_float_complex );

} // void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] )

//-------------------------------------------------------------------------
// compute entries of the observation matrix (N_f_mix x N_points)
//-------------------------------------------------------------------------
__global__ void compute_matrix_kernel( t_float_complex_gpu* d_Phi_float_complex, int N_f_mix, int N_points, t_float_complex_gpu* d_h_ref_float_complex, size_t pitch_h_ref, int* d_indices_grid_FOV_shift_int, size_t pitch_indices_grid_FOV_shift, int index_element, int* d_indices_f_mix_to_sequence, t_float_complex_gpu* d_p_incident_measurement_float_complex, size_t pitch_p_incident_measurement, int* d_indices_f_mix_to_measurement, t_float_complex_gpu* d_prefactors_mix_float_complex, size_t pitch_prefactors_mix, int index_active )
{

	// each thread computes one element in output matrix
	int index_point = blockIdx.x * blockDim.x + threadIdx.x;
	int index_f = blockIdx.y * blockDim.y + threadIdx.y;
	int index_total = index_point * N_f_mix + index_f;

	// multiply aggregated tx group with translation vector
	if( index_f < N_f_mix && index_point < N_points )
	{
		// shift reference spatial transfer function to infer that of the active array element
		int index_point_shift = *( (int*) ( (char *) d_indices_grid_FOV_shift_int + index_element * pitch_indices_grid_FOV_shift ) + index_point );
		
		// compute matrix entry
		d_Phi_float_complex[ index_total ] = cuCmulf( cuCmulf( *( (t_float_complex_gpu*) ( (char *) d_h_ref_float_complex + index_point_shift * pitch_h_ref ) + d_indices_f_mix_to_sequence[ index_f ] ), *( (t_float_complex_gpu*) ( (char *) d_p_incident_measurement_float_complex + index_point * pitch_p_incident_measurement ) + d_indices_f_mix_to_measurement[ index_f ] ) ), *( (t_float_complex_gpu*) ( (char *) d_prefactors_mix_float_complex + index_active * pitch_prefactors_mix ) + index_f ) );
		//d_Phi_float_complex[ index_total ].x = (t_float_gpu) index_total;
		//d_Phi_float_complex[ index_total ].y = (t_float_gpu) index_total;

	} // if( index_f < N_f_mix && index_point < N_points )

} // __global__ void compute_matrix_kernel( t_float_complex_gpu* d_Phi_float_complex, int N_f_mix, int N_points, t_float_complex_gpu* d_h_ref_float_complex, size_t pitch_h_ref, int* d_indices_grid_FOV_shift_int, size_t pitch_indices_grid_FOV_shift, int index_element, int* d_indices_f_mix_to_sequence, t_float_complex_gpu* d_p_incident_measurement_float_complex, size_t pitch_p_incident_measurement, int* d_indices_f_mix_to_measurement, t_float_complex_gpu* d_prefactors_mix_float_complex, size_t pitch_prefactors_mix, int index_active )

//-------------------------------------------------------------------------
// print device status information
//-------------------------------------------------------------------------
void print_device_info( int index_device, int deviceCount, cudaDeviceProp deviceProp )
{
	mexPrintf( " %s\n", "--------------------------------------------------------------------------------" );
	mexPrintf( " Information for GPU device %-1d of %-1d:\n", index_device, deviceCount );
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

//-------------------------------------------------------------------------
// print memory information
//-------------------------------------------------------------------------
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