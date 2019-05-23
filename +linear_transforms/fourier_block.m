%
% compute d-dimensional discrete block fourier transform for various options
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2019-05-17
%
classdef fourier_block < linear_transforms.orthonormal_linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_points_axis

        % dependent properties
        N_dimensions            % number of dimensions
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
        function objects = fourier_block( N_points_axis, size_block )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for N_points_axis
            if ~iscell( N_points_axis )
                N_points_axis = { N_points_axis };
            end

            % ensure cell array for size_block
            if ~iscell( size_block )
                size_block = { size_block };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( N_points_axis, size_block );

            %--------------------------------------------------------------
            % 2.) create d-dimensional discrete block fourier transforms
            %--------------------------------------------------------------
            % total number of lattice points
            N_points = cellfun( @prod, N_points_axis );

            % constructor of superclass
            objects@linear_transforms.orthonormal_linear_transform( N_points, 'fourier_block' );

            % iterate transforms
            for index_object = 1:numel( objects )

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( N_points_axis{ index_object }, size_block{ index_object } );

                % ensure row vectors
                if ~isrow( N_points_axis{ index_object } )
                    errorStruct.message = sprintf( 'N_points_axis{ %d } must be pulse_echo_measurements.setting!', index_object );
                    errorStruct.identifier = 'fourier_block:NoRowVector';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).N_points_axis = N_points_axis{ index_object };

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                % number of dimensions
                objects( index_object ).N_dimensions = numel( objects( index_object ).N_points_axis );

                % number of blocks along each axis
                objects( index_object ).N_blocks_axis = ceil( objects( index_object ).N_points_axis ./ size_block{ index_object } );

                % number of grid points per block
                objects( index_object ).N_points_block = prod( size_block{ index_object } );

                % compute start and stop indices for each block
                objects( index_object ).indices_start = cell( 1, N_dim );
                objects( index_object ).indices_stop = cell( 1, N_dim );
                for index_axis = 1:N_dim

%                     objects( index_object ).indices_start{ index_axis } = ()
                    objects( index_object ).indices_start{ index_axis } = ( ( 1:N_blocks( index_axis ) ) - 1 ) * size_block( index_axis ) + 1;
                    objects( index_object ).indices_stop{ index_axis } = objects( index_object ).indices_start{ index_axis } + size_block( index_axis ) - 1;

                    % check validity of indices
                    if idx_stop{ index_axis }(end) > N_points_axis( index_axis )
                        idx_stop{ index_axis }(end) = N_points_axis( index_axis );
                    end

                end % for index_axis = 1:N_dim

                objects( index_object ).N_points_block_sqrt = sqrt( objects( index_object ).N_points_block );

            end % for index_object = 1:numel( objects )

        end % function objects = fourier_block( N_points_axis, size_block )
       
        %------------------------------------------------------------------
        % overload method: forward transform (forward block DFT)
        %------------------------------------------------------------------
        function y = forward_transform( object, x )

            x = reshape( x, [object.N_points_axis(2), object.N_points_axis(1)] );
            y = zeros( object.N_points_axis(2), object.N_points_axis(1) );
            for index_x = 1:object.N_blocks_axis(1)
                for index_z = 1:object.N_blocks_axis(2)

                    x_block = x( object.indices_start{2}(index_z):object.indices_stop{2}(index_z), object.indices_start{1}(index_x):object.indices_stop{1}(index_x) );
                    y( object.indices_start{2}(index_z):object.indices_stop{2}(index_z), object.indices_start{1}(index_x):object.indices_stop{1}(index_x) ) = fft2( x_block ) / object.N_lattice_block_sqrt;
                end
            end
        end % function y = forward_transform( object, x )

        %------------------------------------------------------------------
        % overload method: adjoint transform (inverse block DFT)
        %------------------------------------------------------------------
        function y = adjoint_transform( LT_fourier_blk, x )

            x = reshape( x, [LT_fourier_blk.N_points_axis(2), LT_fourier_blk.N_points_axis(1)] );
            y = zeros( LT_fourier_blk.N_points_axis(2), LT_fourier_blk.N_points_axis(1) );
            for index_x = 1:LT_fourier_blk.N_blocks_axis(1)
                for index_z = 1:LT_fourier_blk.N_blocks_axis(2)

                    x_block = x( LT_fourier_blk.indices_start{2}(index_z):LT_fourier_blk.indices_stop{2}(index_z), LT_fourier_blk.indices_start{1}(index_x):LT_fourier_blk.indices_stop{1}(index_x) );
                    y( LT_fourier_blk.indices_start{2}(index_z):LT_fourier_blk.indices_stop{2}(index_z), LT_fourier_blk.indices_start{1}(index_x):LT_fourier_blk.indices_stop{1}(index_x) ) = ifft2( x_block ) * LT_fourier_blk.N_lattice_block_sqrt;
                end
            end
        end % function y = adjoint_transform( LT_fourier_blk, x )

    end % methods
    
end % classdef fourier_block < linear_transforms.orthonormal_linear_transform