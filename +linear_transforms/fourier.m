%
% superclass for all d-dimensional discrete Fourier transforms
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2020-01-29
%
classdef fourier < linear_transforms.linear_transform_vector

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_points_axis ( 1, : ) double { mustBePositive, mustBeInteger, mustBeNonempty } = [ 512, 512 ]

        % dependent properties
        N_dimensions ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 2
        N_points_sqrt ( 1, 1 ) double { mustBePositive, mustBeNonempty } = 512

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = fourier( N_points_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for N_points_axis
            if ~iscell( N_points_axis )
                N_points_axis = { N_points_axis };
            end

            %--------------------------------------------------------------
            % 2.) create discrete Fourier transforms
            %--------------------------------------------------------------
            % compute numbers of points and their square roots
            N_points = cellfun( @( x ) prod( x( : ) ), N_points_axis );
            N_points_sqrt = sqrt( N_points );

            % constructor of superclass
            objects@linear_transforms.linear_transform_vector( N_points, N_points );

            % iterate discrete Fourier transforms
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).N_points_axis = N_points_axis{ index_object };

                % set dependent properties
                objects( index_object ).N_dimensions = numel( objects( index_object ).N_points_axis );
                objects( index_object ).N_points_sqrt = N_points_sqrt( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = fourier( N_points_axis )

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

            % calling function ensures numeric column vector for x
            % calling function ensures equal numbers of points

            %--------------------------------------------------------------
            % 2.) compute forward Fourier transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            % apply forward transform
            y_act = fftn( x ) / LT.N_points_sqrt;

            % return result as column vector
            y = y_act( : );

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

            % calling function ensures numeric column vector for x
            % calling function ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint Fourier transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            % apply adjoint transform
            y_act = ifftn( x ) * LT.N_points_sqrt;

            % return result as column vector
            y = y_act( : );

        end % function y = adjoint_transform_vector( LT, x )

        %------------------------------------------------------------------
        % display coefficients (single vector)
        %------------------------------------------------------------------
        function display_coefficients_vector( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class linear_transforms.linear_transform_vector (scalar) for LT
            % calling function ensures numeric column vector for x
            % calling function ensures equal numbers of coefficients

            % ensure class linear_transforms.fourier (scalar)
            if ~( isa( LT, 'linear_transforms.fourier' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.fourier!';
                errorStruct.identifier = 'display_coefficients_vector:NoSingleFourierTransform';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) display coefficients (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            % logarithmic compression
            x_dB = fftshift( illustration.dB( x, 10 )', 2 );

            % display vector
            if LT.N_dimensions == 2
                imagesc( x_dB, [ -60, 0 ] );
                title( 'Fourier coefficients' );
                colorbar;
            end

        end % function display_coefficients_vector( LT, x )

	end % methods (Access = protected, Hidden)

end % classdef fourier < linear_transforms.linear_transform_vector
