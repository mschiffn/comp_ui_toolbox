%
% compute d-dimensional discrete fourier transform for various options
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2019-02-23
%
classdef fourier < linear_transforms.orthonormal_linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        N_dimensions
        N_lattice_axis
        N_lattice_sqrt
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT_fourier = fourier( N_lattice_axis )

            % total number of lattice points
            N_dim = numel( N_lattice_axis );
            N_lattice = prod( N_lattice_axis );

            % constructor of superclass
            LT_fourier@linear_transforms.orthonormal_linear_transform( N_lattice, 'fourier' );

            % internal properties
            LT_fourier.N_dimensions = N_dim;
            LT_fourier.N_lattice_axis = N_lattice_axis;
            LT_fourier.N_lattice_sqrt = sqrt( N_lattice );
        end

        %------------------------------------------------------------------
        % overload method: forward transform (forward DFT)
        %------------------------------------------------------------------
        function y = forward_transform( LT_fourier, x )

            x = reshape( x, [LT_fourier.N_lattice_axis(2), LT_fourier.N_lattice_axis(1)] );
            y = fftn( x ) / LT_fourier.N_lattice_sqrt;

        end

        %------------------------------------------------------------------
        % overload method: adjoint transform (inverse DFT)
        %------------------------------------------------------------------
        function y = adjoint_transform( LT_fourier, x )

            x = reshape( x, [LT_fourier.N_lattice_axis(2), LT_fourier.N_lattice_axis(1)] );
            y = ifft2( x ) * LT_fourier.N_lattice_sqrt;

        end

    end % methods
    
end % classdef fourier < linear_transforms.orthonormal_linear_transform
