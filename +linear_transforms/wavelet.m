%
% compute two-dimensional discrete wavelet transform for various options
% author: Martin Schiffner
% date: 2016-08-13
%
classdef wavelet < linear_transforms.orthonormal_linear_transform
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        wavelet_name
        wavelet_parameter
        N_points
        scale_coarsest
        quadrature_mirror_filter
    end % properties
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT_wavelet = wavelet( wavelet_name, wavelet_parameter, N_points, scale_coarsest )

            % total number of lattice points
            N_lattice = N_points^2;

            % create name string
            str_name = sprintf( 'wavelet_%s', wavelet_name );

            if strcmp( wavelet_name, 'Haar' ) ~= 1

                str_name = sprintf( '%s_%d', str_name, wavelet_parameter );
            end

            str_name = sprintf( '%s_%d', str_name, scale_coarsest );

            % constructor of superclass
            LT_wavelet@linear_transforms.orthonormal_linear_transform( N_lattice, str_name );

            % internal properties
            LT_wavelet.wavelet_name         = wavelet_name;
            LT_wavelet.wavelet_parameter	= wavelet_parameter;
            LT_wavelet.N_points             = N_points;
            LT_wavelet.scale_coarsest       = scale_coarsest;

            % compute quadrature mirror filter (QMF)
            LT_wavelet.quadrature_mirror_filter = MakeONFilter( wavelet_name, wavelet_parameter );
        end

        %------------------------------------------------------------------
        % overload method: forward transform (forward DWT)
        %------------------------------------------------------------------
        function y = forward_transform( LT_wavelet, x )

            % transform real and imaginary parts separately
            x = reshape( x, [LT_wavelet.N_points, LT_wavelet.N_points] );
            y = FWT2_PO( real( x ), LT_wavelet.scale_coarsest, LT_wavelet.quadrature_mirror_filter );
            y = y + 1j * FWT2_PO( imag( x ), LT_wavelet.scale_coarsest, LT_wavelet.quadrature_mirror_filter );
        end

        %------------------------------------------------------------------
        % overload method: adjoint transform (inverse DWT)
        %------------------------------------------------------------------
        function y = adjoint_transform( LT_wavelet, x )

            % transform real and imaginary parts separately
            x = reshape( x, [LT_wavelet.N_points, LT_wavelet.N_points] );
            y = IWT2_PO( real( x ), LT_wavelet.scale_coarsest, LT_wavelet.quadrature_mirror_filter );
            y = y + 1j * IWT2_PO( imag( x ), LT_wavelet.scale_coarsest, LT_wavelet.quadrature_mirror_filter );
        end

    end % methods
    
end % classdef wavelet