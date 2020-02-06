%
% compute d-dimensional discrete block fourier transform
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2020-01-31
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
        N_points_block
        N_points_block_sqrt
        indices_start
        indices_stop

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
            % ensure cell array for N_points_axis
            if ~iscell( N_points_axis )
                N_points_axis = { N_points_axis };
            end

            % ensure cell array for N_points_block_axis
            if ~iscell( N_points_block_axis )
                N_points_block_axis = { N_points_block_axis };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( N_points_axis, N_points_block_axis );

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
                % a) set independent properties
                %----------------------------------------------------------
                % number of points along each axis
                objects( index_object ).N_points_axis = N_points_axis{ index_object };

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                % number of dimensions
                objects( index_object ).N_dimensions = numel( objects( index_object ).N_points_axis );

                % number of blocks along each axis
                objects( index_object ).N_blocks_axis = ceil( objects( index_object ).N_points_axis ./ N_points_block_axis{ index_object } );

                % number of grid points per block
                objects( index_object ).N_points_block = N_points_block( index_object );

                % compute start and stop indices for each block
                objects( index_object ).indices_start = cell( 1, N_dim );
                objects( index_object ).indices_stop = cell( 1, N_dim );
                for index_axis = 1:N_dim

%                     objects( index_object ).indices_start{ index_axis } = ()
                    objects( index_object ).indices_start{ index_axis } = ( ( 1:N_blocks( index_axis ) ) - 1 ) * N_points_block_axis( index_axis ) + 1;
                    objects( index_object ).indices_stop{ index_axis } = objects( index_object ).indices_start{ index_axis } + N_points_block_axis( index_axis ) - 1;

                    % check validity of indices
                    if idx_stop{ index_axis }(end) > N_points_axis( index_axis )
                        idx_stop{ index_axis }(end) = N_points_axis( index_axis );
                    end

                end % for index_axis = 1:N_dim

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
            % ensure class linear_transforms.fourier (scalar)
            if ~( isa( LT, 'linear_transforms.fourier' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.fourier!';
                errorStruct.identifier = 'forward_transform_vector:NoSingleFourierTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward Fourier transform (single vector)
            %--------------------------------------------------------------
            % create size matrix
            size_matrix = [ LT.N_points_block_axis; LT.N_blocks_axis ];

            % block partitioning
            x = reshape( x, size_matrix( : )' );
            
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            
            mat2cell( x, LT.N_blocks_axis )
            x = reshape( x, [object.N_points_axis(2), object.N_points_axis(1)] );
            y = zeros( object.N_points_axis(2), object.N_points_axis(1) );
            for index_x = 1:object.N_blocks_axis(1)
                for index_z = 1:object.N_blocks_axis(2)

                    x_block = x( object.indices_start{2}(index_z):object.indices_stop{2}(index_z), object.indices_start{1}(index_x):object.indices_stop{1}(index_x) );
                    y( object.indices_start{2}(index_z):object.indices_stop{2}(index_z), object.indices_start{1}(index_x):object.indices_stop{1}(index_x) ) = fft2( x_block ) / object.N_lattice_block_sqrt;
                end
            end

        end % function y = forward_transform_vector( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single vector)
        %------------------------------------------------------------------
        function y = adjoint_transform_vector( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.fourier (scalar)
            if ~( isa( LT, 'linear_transforms.fourier' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.fourier!';
                errorStruct.identifier = 'adjoint_transform_vector:NoSingleFourierTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint Fourier transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            x = reshape( x, [LT_fourier_blk.N_points_axis(2), LT_fourier_blk.N_points_axis(1)] );
            y = zeros( LT_fourier_blk.N_points_axis(2), LT_fourier_blk.N_points_axis(1) );
            for index_x = 1:LT_fourier_blk.N_blocks_axis(1)
                for index_z = 1:LT_fourier_blk.N_blocks_axis(2)

                    x_block = x( LT_fourier_blk.indices_start{2}(index_z):LT_fourier_blk.indices_stop{2}(index_z), LT_fourier_blk.indices_start{1}(index_x):LT_fourier_blk.indices_stop{1}(index_x) );
                    y( LT_fourier_blk.indices_start{2}(index_z):LT_fourier_blk.indices_stop{2}(index_z), LT_fourier_blk.indices_start{1}(index_x):LT_fourier_blk.indices_stop{1}(index_x) ) = ifft2( x_block ) * LT_fourier_blk.N_lattice_block_sqrt;
                end
            end

        end % function y = adjoint_transform_vector( LT, x )

	end % methods (Access = protected, Hidden)

end % classdef fourier_block < linear_transforms.linear_transform_vector
