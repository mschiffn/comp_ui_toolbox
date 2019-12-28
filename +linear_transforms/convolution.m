%
% compute discrete convolution with specified kernel
%
% author: Martin F. Schiffner
% date: 2019-12-08
% modified: 2019-12-27
%
classdef convolution < linear_transforms.linear_transform

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
            objects@linear_transforms.linear_transform( N_coefficients, N_points );

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

        %------------------------------------------------------------------
        % forward transform (overload forward_transform method)
        %------------------------------------------------------------------
        function [ y_dft, y_mat ] = forward_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for x
            if ~iscell( x )
                x = { x };
            end

            % multiple LTs / single x
            if ~isscalar( LTs ) && isscalar( x )
                x = repmat( x, size( LTs ) );
            end

            % single LTs / multiple x
            if isscalar( LTs ) && ~isscalar( x )
                x = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) compute forward convolutions
            %--------------------------------------------------------------
% TODO: choose computation method automatically
            % specify cell array for y
            y_dft = cell( size( LTs ) );
            y_mat = cell( size( LTs ) );

            % iterate convolutions
            for index_object = 1:numel( LTs )

                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'forward_transform:NoNumericMatrix';
                    error( errorStruct );
                end

                % apply forward transform using DFT
                y_dft{ index_object } = ifft( LTs( index_object ).kernel_dft .* fft( x{ index_object }, LTs( index_object ).N_dft ) );
                if LTs( index_object ).cut_off
                    y_dft{ index_object } = y_dft{ index_object }( ( LTs( index_object ).M_kernel + 1 ):( end - LTs( index_object ).M_kernel ), : );
                end

                % apply forward transform using matrix
                if nargout > 1
                    y_mat{ index_object } = LTs( index_object ).matrix * x{ index_object };
                end

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
            if isscalar( LTs )
                y_dft = y_dft{ 1 };
                y_mat = y_mat{ 1 };
            end

        end % function [ y_dft, y_mat ] = forward_transform( LTs, x )

        %------------------------------------------------------------------
        % adjoint transform (overload adjoint_transform method)
        %------------------------------------------------------------------
        function [ y_dft, y_mat ] = adjoint_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for x
            if ~iscell( x )
                x = { x };
            end

            % multiple LTs / single x
            if ~isscalar( LTs ) && isscalar( x )
                x = repmat( x, size( LTs ) );
            end

            % single LTs / multiple x
            if isscalar( LTs ) && ~isscalar( x )
                x = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) compute adjoint convolutions
            %--------------------------------------------------------------
            % specify cell array for y
            y_dft = cell( size( LTs ) );
            y_mat = cell( size( LTs ) );

            % iterate convolutions
            for index_object = 1:numel( LTs )

                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'adjoint_transform:NoNumericMatrix';
                    error( errorStruct );
                end

                % apply adjoint transform using matrix
                if nargout > 1
                    y_mat{ index_object } = LTs( index_object ).matrix_adj * x{ index_object };
                end

                % apply adjoint transform using DFT
                x{ index_object } = [ zeros( LTs( index_object ).M_kernel, size( x{ index_object }, 2 ) ); x{ index_object }; zeros( LTs( index_object ).M_kernel, size( x{ index_object }, 2 ) ) ];
                y_dft{ index_object } = ifft( LTs( index_object ).kernel_dft_conj .* fft( x{ index_object }, LTs( index_object ).N_dft ) );
                y_dft{ index_object } = y_dft{ index_object }( 1:LTs( index_object ).N_points, : );

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
            if isscalar( LTs )
                y_dft = y_dft{ 1 };
                y_mat = y_mat{ 1 };
            end

        end % function [ y_dft, y_mat ] = adjoint_transform( LTs, x )

    end % methods

end % classdef convolution < linear_transforms.linear_transform
