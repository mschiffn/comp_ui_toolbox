%
% superclass for all fft-based discrete convolutions
%
% author: Martin F. Schiffner
% date: 2020-04-02
% modified: 2020-04-03
%
classdef fft < linear_transforms.convolutions.convolution

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % dependent properties
        N_dft ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 1 % order of the DFT
        kernel_dft ( :, 1 )
        kernel_dft_conj ( :, 1 )

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = fft( kernels, N_points, cut_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures cell array for kernels
            % superclass ensures column vectors for kernels
            % superclass ensures nonempty positive integers for N_points

            % ensure nonempty cut_off
            if nargin < 3 || isempty( cut_off )
                cut_off = true;
            end

            % property validation function ensures logical for cut_off

            %--------------------------------------------------------------
            % 2.) create fft-based discrete convolutions
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.convolutions.convolution( kernels, N_points, cut_off );

            % iterate fft-based discrete convolutions
            for index_object = 1:numel( objects )

                % set dependent properties
                objects( index_object ).N_dft = numel( objects( index_object ).kernel ) + objects( index_object ).N_points - 1;
                objects( index_object ).kernel_dft = fft( objects( index_object ).kernel, objects( index_object ).N_dft );
                objects( index_object ).kernel_dft_conj = conj( objects( index_object ).kernel_dft );

            end % for index_object = 1:numel( objects )

        end % function objects = fft( kernels, N_points, cut_off )

    end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % forward transform (single matrix)
        %------------------------------------------------------------------
        function y = forward_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.convolutions.fft (scalar)
            if ~( isa( LT, 'linear_transforms.convolutions.fft' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.convolutions.fft!';
                errorStruct.identifier = 'forward_transform_matrix:NoSingleFFTConvolution';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward convolutions (single matrix)
            %--------------------------------------------------------------
            % apply forward transform using DFT
            y = ifft( LT.kernel_dft .* fft( x, LT.N_dft ) );
            if LT.cut_off
                y = y( ( LT.M_kernel + 1 ):( end - LT.M_kernel ), : );
            end

        end % function y = forward_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single matrix)
        %------------------------------------------------------------------
        function y = adjoint_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.convolutions.fft (scalar)
            if ~( isa( LT, 'linear_transforms.convolutions.fft' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.convolutions.fft!';
                errorStruct.identifier = 'adjoint_transform_matrix:NoSingleFFTConvolution';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint convolutions (single matrix)
            %--------------------------------------------------------------
            % apply adjoint transform using DFT
            if LT.cut_off
                x = [ zeros( LT.M_kernel, size( x, 2 ) ); x; zeros( LT.M_kernel, size( x, 2 ) ) ];
            end
            y = ifft( LT.kernel_dft_conj .* fft( x, LT.N_dft ) );
            y = y( 1:LT.N_points, : );

        end % function y = adjoint_transform_matrix( LT, x )

	end % methods (Access = protected, Hidden)

end % classdef matrix < linear_transforms.convolutions.convolution
