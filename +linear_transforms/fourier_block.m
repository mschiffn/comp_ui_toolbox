%
% compute d-dimensional discrete block fourier transform for various options
% author: Martin Schiffner
% date: 2016-08-12
%
classdef fourier_block < linear_transforms.orthonormal_linear_transform
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        N_dimensions
        N_lattice_axis
        N_blocks_axis
        N_lattice_block
        N_lattice_block_sqrt
        indices_start
        indices_stop
    end % properties
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT_fourier_blk = fourier_block( N_lattice_axis, size_block )

            % total number of lattice points
            N_dim = numel( N_lattice_axis );
            N_lattice = prod( N_lattice_axis );

            % constructor of superclass
            LT_fourier_blk@linear_transforms.orthonormal_linear_transform( N_lattice, 'fourier_block' );

            % number of blocks along each axis
            N_blocks = ceil( N_lattice_axis ./ size_block );
            
            % number of lattice points per block
            N_block	 = prod( size_block );

            % compute start and stop indices for each block
            idx_start = cell( 1, N_dim );
            idx_stop  = cell( 1, N_dim );
            for index_axis = 1:N_dim
                                
                idx_start{ index_axis }	= ( (1:N_blocks( index_axis )) - 1 ) * size_block( index_axis ) + 1;
                idx_stop{ index_axis }	= idx_start{ index_axis } + size_block( index_axis ) - 1;

                % check validity of indices
                if idx_stop{ index_axis }(end) > N_lattice_axis( index_axis )
                    idx_stop{ index_axis }(end) = N_lattice_axis( index_axis );
                end
            end

            % internal properties
            LT_fourier_blk.N_dimensions = N_dim;
            LT_fourier_blk.N_lattice_axis = N_lattice_axis;
            LT_fourier_blk.N_lattice_block = N_block;
            LT_fourier_blk.N_lattice_block_sqrt = sqrt( N_block );
            LT_fourier_blk.N_blocks_axis	= N_blocks;
            LT_fourier_blk.indices_start	= idx_start;
            LT_fourier_blk.indices_stop     = idx_stop;

        end
       
        %------------------------------------------------------------------
        % overload method: forward transform (forward block DFT)
        %------------------------------------------------------------------
        function y = forward_transform( LT_fourier_blk, x )

            x = reshape( x, [LT_fourier_blk.N_lattice_axis(2), LT_fourier_blk.N_lattice_axis(1)] );
            y = zeros( LT_fourier_blk.N_lattice_axis(2), LT_fourier_blk.N_lattice_axis(1) );
            for index_x = 1:LT_fourier_blk.N_blocks_axis(1)
                for index_z = 1:LT_fourier_blk.N_blocks_axis(2)

                    x_block = x( LT_fourier_blk.indices_start{2}(index_z):LT_fourier_blk.indices_stop{2}(index_z), LT_fourier_blk.indices_start{1}(index_x):LT_fourier_blk.indices_stop{1}(index_x) );
                    y( LT_fourier_blk.indices_start{2}(index_z):LT_fourier_blk.indices_stop{2}(index_z), LT_fourier_blk.indices_start{1}(index_x):LT_fourier_blk.indices_stop{1}(index_x) ) = fft2( x_block ) / LT_fourier_blk.N_lattice_block_sqrt;
                end
            end
        end

        %------------------------------------------------------------------
        % overload method: adjoint transform (inverse block DFT)
        %------------------------------------------------------------------
        function y = adjoint_transform( LT_fourier_blk, x )

            x = reshape( x, [LT_fourier_blk.N_lattice_axis(2), LT_fourier_blk.N_lattice_axis(1)] );
            y = zeros( LT_fourier_blk.N_lattice_axis(2), LT_fourier_blk.N_lattice_axis(1) );
            for index_x = 1:LT_fourier_blk.N_blocks_axis(1)
                for index_z = 1:LT_fourier_blk.N_blocks_axis(2)

                    x_block = x( LT_fourier_blk.indices_start{2}(index_z):LT_fourier_blk.indices_stop{2}(index_z), LT_fourier_blk.indices_start{1}(index_x):LT_fourier_blk.indices_stop{1}(index_x) );
                    y( LT_fourier_blk.indices_start{2}(index_z):LT_fourier_blk.indices_stop{2}(index_z), LT_fourier_blk.indices_start{1}(index_x):LT_fourier_blk.indices_stop{1}(index_x) ) = ifft2( x_block ) * LT_fourier_blk.N_lattice_block_sqrt;
                end
            end
        end

    end % methods
    
end % classdef fourier_block < linear_transforms.orthonormal_linear_transform