function nvmex_linux( varargin )
% compile CUDA code using MATLAB mex interface
%
% author: Martin F. Schiffner
% date: 2011-10-01
% modified: 2019-07-02

%--------------------------------------------------------------------------
% 1.) set paths
%--------------------------------------------------------------------------
% root path
CUDA_ROOT = '/usr/local/cuda';

% dependent paths
MEX_INC_PATH  = sprintf( '%s/extern/include', matlabroot );
CUDA_BIN_PATH = sprintf( '%s/bin', CUDA_ROOT );
CUDA_INC_PATH = sprintf( '%s/include', CUDA_ROOT );
CUDA_LIB_PATH = sprintf( '%s/lib64', CUDA_ROOT );

% specify architecture
% (embed PTX code for several architectures if necessary; PTX code will be compiled just-in-time)
CUDA_ARCH = 'sm_30';
% -use_fast_math

%--------------------------------------------------------------------------
% 2.) check arguments
%--------------------------------------------------------------------------
files = struct( 'path', cell( nargin, 1 ), 'filename', cell( nargin, 1 ), 'ext', cell( nargin, 1 ), 'intermediate', cell( nargin, 1 ) );
for index_arg = 1:numel( varargin )

	[ files( index_arg ).path, files( index_arg ).filename, files( index_arg ).ext ] = fileparts( varargin{ index_arg } );
	files( index_arg ).intermediate = fullfile( files( index_arg ).path, strcat( files( index_arg ).filename, '.cpp' ) );
    files( index_arg ).result = fullfile( files( index_arg ).path, files( index_arg ).filename );

end % for index_arg = 1:numel( varargin )

%--------------------------------------------------------------------------
% 2.) create command strings
%--------------------------------------------------------------------------
nvcc_command_string = sprintf( '%s/nvcc -Xptxas="-v" --cuda --disable-warnings -arch=%s -I"%s" -I"%s" %s -o %s', CUDA_BIN_PATH, CUDA_ARCH, CUDA_INC_PATH, MEX_INC_PATH, varargin{ 1 }, files( index_arg ).intermediate );
mex_command_string = sprintf( '%s/bin/mex -R2018a -v %s -L"%s" -lcudart -lcublas -output %s', matlabroot, files( index_arg ).intermediate, CUDA_LIB_PATH, files( index_arg ).result );
clean_command_string = sprintf( 'rm %s', files( index_arg ).intermediate );

%--------------------------------------------------------------------------
% 3.) execute commands
%--------------------------------------------------------------------------
status = system( nvcc_command_string );
if status == 0
	system( mex_command_string );
    system( clean_command_string );
end
