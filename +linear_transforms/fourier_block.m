%
% compute d-dimensional discrete block fourier transform
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2020-10-20
%
classdef fourier_block < linear_transforms.linear_transform_vector

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_points_axis ( 1, : ) double { mustBePositive, mustBeInteger, mustBeNonempty } = [ 512, 512 ]

        % dependent properties
        N_dimensions ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 2  % number of dimensions
        N_blocks_axis           % number of blocks along each axis
        N_blocks
        N_points_block
        N_points_block_sqrt
        indices_start
        indices_stop
% TODO: partitioning cell array size( N_blocks_axis ) content N_points_block_axis
        partitioning

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = fourier_block( N_points_axis, N_points_block_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure cell array for N_points_axis
            if ~iscell( N_points_axis )
                N_points_axis = { N_points_axis };
            end

            % ensure cell array for N_points_block_axis
            if ~iscell( N_points_block_axis )
                N_points_block_axis = { N_points_block_axis };
            end

            % ensure equal number of dimensions and sizes
            [ N_points_axis, N_points_block_axis ] = auxiliary.ensureEqualSize( N_points_axis, N_points_block_axis );

            %--------------------------------------------------------------
            % 2.) create d-dimensional discrete block fourier transforms
            %--------------------------------------------------------------
            % compute numbers of points and their square roots
            N_points = cellfun( @( x ) prod( x( : ) ), N_points_axis );

            % compute numbers of points per block and their square roots
            N_points_block = cellfun( @( x ) prod( x( : ) ), N_points_block_axis );
            N_points_block_sqrt = sqrt( N_points_block );

            % constructor of superclass
            objects@linear_transforms.linear_transform_vector( N_points, N_points );

            % iterate transforms
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( N_points_axis{ index_object }, N_points_block_axis{ index_object } );

                % ensure row vectors
                if ~isrow( N_points_axis{ index_object } )
                    errorStruct.message = sprintf( 'N_points_axis{ %d } must be a row vector!', index_object );
                    errorStruct.identifier = 'fourier_block:NoRowVector';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) set independent properties
                %----------------------------------------------------------
                % number of points along each axis
                objects( index_object ).N_points_axis = N_points_axis{ index_object };

                %----------------------------------------------------------
                % c) set dependent properties
                %----------------------------------------------------------
                % number of dimensions
                objects( index_object ).N_dimensions = numel( objects( index_object ).N_points_axis );

                % number of blocks along each axis
                objects( index_object ).N_blocks_axis = ceil( objects( index_object ).N_points_axis ./ N_points_block_axis{ index_object } );
                objects( index_object ).N_blocks = prod( objects( index_object ).N_blocks_axis );

                % number of grid points per block
                objects( index_object ).N_points_block = N_points_block( index_object );

                % compute partitioning
                objects( index_object ).partitioning = cell( 1, objects( index_object ).N_dimensions );
                for index_axis = 1:objects( index_object ).N_dimensions
                    objects( index_object ).partitioning{ index_axis } = repmat( N_points_block_axis{ index_object }( index_axis ), [ 1, objects( index_object ).N_blocks_axis( index_axis ) ] );
                end % for index_axis = 1:objects( index_object ).N_dimensions

                %
                objects( index_object ).N_points_block_sqrt = N_points_block_sqrt( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = fourier_block( N_points_axis, N_points_block_axis )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % forward transform (single vector)
        %------------------------------------------------------------------
        function y = forward_transform_vector( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.fourier_block (scalar)
            if ~( isa( LT, 'linear_transforms.fourier_block' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.fourier_block!';
                errorStruct.identifier = 'forward_transform_vector:NoSingleBlockFourierTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward block Fourier transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            % block partitioning
            x = mat2cell( x, LT.partitioning{ : } );

            % specify cell array for y
            y = cell( LT.N_blocks_axis );

            % iterate blocks
            for index_block = 1:LT.N_blocks

                % apply forward block transform
                y{ index_block } = fftn( x{ index_block } ) / LT.N_points_block_sqrt;

            end % for index_block = 1:LT.N_blocks

            % avoid cell array for y
            y = reshape( cell2mat( y ), [ LT.N_points, 1 ] );

        end % function y = forward_transform_vector( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single vector)
        %------------------------------------------------------------------
        function y = adjoint_transform_vector( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.fourier_block (scalar)
            if ~( isa( LT, 'linear_transforms.fourier_block' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.fourier_block!';
                errorStruct.identifier = 'adjoint_transform_vector:NoSingleBlockFourierTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint block Fourier transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            % block partitioning
            x = mat2cell( x, LT.partitioning{ : } );

            % specify cell array for y
            y = cell( LT.N_blocks_axis );

            % iterate blocks
            for index_block = 1:LT.N_blocks

                % apply inverse block transform
                y{ index_block } = ifftn( x{ index_block } ) * LT.N_points_block_sqrt;

            end % for index_block = 1:LT.N_blocks

            % avoid cell array for y
            y = reshape( cell2mat( y ), [ LT.N_points, 1 ] );

        end % function y = adjoint_transform_vector( LT, x )

        %------------------------------------------------------------------
        % display coefficients (single vector)
        %------------------------------------------------------------------
        function display_coefficients_vector( LT, x, dynamic_range_dB, factor_dB )

            

        end % function display_coefficients_vector( LT, x, dynamic_range_dB, factor_dB )

        %------------------------------------------------------------------
        % relative RMSEs of best s-sparse approximations (single vector)
        %------------------------------------------------------------------
        function [ rel_RMSEs, axis_s ] = rel_RMSE_vector( LT, y )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            %--------------------------------------------------------------
            % 2.) compute relative RMSEs of best s-sparse approximations (single vector)
            %--------------------------------------------------------------
            % sort absolute values of transform coefficients (ascending order)
            y_abs_sorted = sort( abs( y ), 1, 'ascend' );

            % determine relative root mean-squared approximation error
            rel_RMSEs = flip( sqrt( cumsum( y_abs_sorted.^2 ) ) / norm( y, 2 ) );

            % number of coefficients corresponding to relative RMSE
            axis_s = ( 0:( LT.N_coefficients - 1 ) );

        end % function [ rel_RMSEs, axis_s ] = rel_RMSE_vector( LT, y )

	end % methods (Access = protected, Hidden)

end % classdef fourier_block < linear_transforms.linear_transform_vector
