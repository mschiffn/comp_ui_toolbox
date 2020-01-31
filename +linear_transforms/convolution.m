%
% compute discrete convolution with specified kernel
%
% author: Martin F. Schiffner
% date: 2019-12-08
% modified: 2020-01-29
%
classdef convolution < linear_transforms.linear_transform_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % independent properties
        kernel ( :, 1 )
        cut_off ( 1, 1 ) logical { mustBeNonempty } = true                          % cut off results to ensure square matrix

        % dependent properties
        M_kernel ( 1, 1 ) double { mustBePositive, mustBeNonempty } = 1             % symmetric length of kernel
        N_dft ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 1 % order of the DFT
        kernel_dft ( :, 1 )
        kernel_dft_conj ( :, 1 )
        matrix
        matrix_adj

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = convolution( kernels, N_points, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for kernels
            if ~iscell( kernels )
                kernels = { kernels };
            end

            % ensure column vectors for kernels
            indicator = cellfun( @( x ) ~iscolumn( x ), kernels );
            if any( indicator( : ) )
                errorStruct.message = 'kernels must contain column vectors!';
                errorStruct.identifier = 'sequence_increasing:NoPhysicalQuantity';
                error( errorStruct );
            end

            % superclass ensures nonempty positive integers for N_points

            % ensure nonempty cut_off
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                cut_off = varargin{ 1 };
            else
                cut_off = true( size( kernels ) );
            end

            % property validation function ensures logical for cut_off

            % multiple kernels / single N_points
            if ~isscalar( kernels ) && isscalar( N_points )
                N_points = repmat( N_points, size( kernels ) );
            end

            % multiple kernels / single cut_off
            if ~isscalar( kernels ) && isscalar( cut_off )
                cut_off = repmat( cut_off, size( kernels ) );
            end

            % multiple N_points / single kernels
            if ~isscalar( N_points ) && isscalar( kernels )
                kernels = repmat( kernels, size( N_points ) );
            end

            % multiple N_points / single cut_off
            if ~isscalar( N_points ) && isscalar( cut_off )
                cut_off = repmat( cut_off, size( N_points ) );
            end

            % multiple cut_off / single kernels
            if ~isscalar( cut_off ) && isscalar( kernels )
                kernels = repmat( kernels, size( cut_off ) );
            end

            % multiple cut_off / single N_points
            if ~isscalar( cut_off ) && isscalar( N_points )
                N_points = repmat( N_points, size( cut_off ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( kernels, N_points, cut_off );

            %--------------------------------------------------------------
            % 2.) create convolutions
            %--------------------------------------------------------------
            % symmetric lengths of kernels
            M_kernel = ( cellfun( @numel, kernels ) - 1 ) / 2;

            % ensure integers if cut_off are true
            mustBeInteger( cut_off .* M_kernel );

            % orders of the DFTs
            N_dft = cellfun( @numel, kernels ) + N_points - 1;

            % number of coefficients
            N_coefficients = ~cut_off .* ( cellfun( @numel, kernels ) - 1 ) + N_points;

            % constructor of superclass
            objects@linear_transforms.linear_transform_matrix( N_coefficients, N_points );

            % iterate convolutions
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).kernel = kernels{ index_object };
                objects( index_object ).cut_off = cut_off( index_object );

                % set dependent properties
                objects( index_object ).M_kernel = M_kernel( index_object );
                objects( index_object ).N_dft = N_dft( index_object );
                objects( index_object ).kernel_dft = fft( objects( index_object ).kernel, objects( index_object ).N_dft );
                objects( index_object ).kernel_dft_conj = conj( objects( index_object ).kernel_dft );

                objects( index_object ).matrix = convmtx( objects( index_object ).kernel, objects( index_object ).N_points );
                if objects( index_object ).cut_off
                    objects( index_object ).matrix = objects( index_object ).matrix( ( objects( index_object ).M_kernel + 1 ):( end - objects( index_object ).M_kernel ), : );
                end
                objects( index_object ).matrix_adj = objects( index_object ).matrix';

            end % for index_object = 1:numel( objects )

        end % function objects = convolution( kernels, N_points, varargin )

    end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % forward transform (single matrix)
        %------------------------------------------------------------------
        function [ y_dft, y_mat ] = forward_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.convolution (scalar)
            if ~( isa( LT, 'linear_transforms.convolution' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.convolution!';
                errorStruct.identifier = 'forward_transform_matrix:NoSingleConvolution';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward convolutions (single matrix)
            %--------------------------------------------------------------
            % apply forward transform using DFT
            y_dft = ifft( LT.kernel_dft .* fft( x, LT.N_dft ) );
            if LT.cut_off
                y_dft = y_dft( ( LT.M_kernel + 1 ):( end - LT.M_kernel ), : );
            end

            % apply forward transform using matrix
            if nargout > 1
                y_mat = LT.matrix * x;
            end

        end % function [ y_dft, y_mat ] = forward_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single matrix)
        %------------------------------------------------------------------
        function [ y_dft, y_mat ] = adjoint_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.convolution (scalar)
            if ~( isa( LT, 'linear_transforms.convolution' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.convolution!';
                errorStruct.identifier = 'adjoint_transform_matrix:NoSingleConvolution';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint convolutions (single matrix)
            %--------------------------------------------------------------
            % apply adjoint transform using matrix
            if nargout > 1
                y_mat = LT.matrix_adj * x;
            end

            % apply adjoint transform using DFT
            x = [ zeros( LT.M_kernel, size( x, 2 ) ); x; zeros( LT.M_kernel, size( x, 2 ) ) ];
            y_dft = ifft( LT.kernel_dft_conj .* fft( x, LT.N_dft ) );
            y_dft = y_dft( 1:LT.N_points, : );

        end % function [ y_dft, y_mat ] = adjoint_transform_matrix( LT, x )

	end % methods (Access = protected, Hidden)

end % classdef convolution < linear_transforms.linear_transform_matrix
