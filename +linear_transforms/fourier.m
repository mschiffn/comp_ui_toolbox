%
% superclass for all d-dimensional discrete Fourier transforms
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2020-08-06
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
            % ensure two arguments
            narginchk( 1, 1 );

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
        function display_coefficients_vector( LT, x, dynamic_range_dB, factor_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure at least two and at most four arguments
            narginchk( 2, 4 );

            % calling function ensures class linear_transforms.linear_transform_vector (scalar) for LT
            % calling function ensures numeric column vector for x
            % calling function ensures equal numbers of coefficients

            % ensure class linear_transforms.fourier (scalar)
            if ~( isa( LT, 'linear_transforms.fourier' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be a single linear_transforms.fourier!';
                errorStruct.identifier = 'display_coefficients_vector:NoSingleFourierTransform';
                error( errorStruct );
            end

            % ensure nonempty dynamic_range_dB
            if nargin < 3 || isempty( dynamic_range_dB )
                dynamic_range_dB = 60;
            end

            % ensure nonempty factor_dB
            if nargin < 4 || isempty( factor_dB )
                factor_dB = 20;
            end

            %--------------------------------------------------------------
            % 2.) display coefficients (single vector)
            %--------------------------------------------------------------
            % logarithmic compression
            x_dB = illustration.dB( x, factor_dB );

            % check dimensionality of the discrete Fourier transform
            if LT.N_dimensions == 1

                % direct display of column vector for one-dimensional discrete Fourier transforms
                index_shift = ceil( LT.N_points_axis / 2 );
                plot( ( ( index_shift - LT.N_points_axis ):( index_shift - 1 ) ), fftshift( x_dB ) );

            else

                % reshape column vector into array and remove dimensions of length 1
                x_dB = squeeze( reshape( x_dB, LT.N_points_axis ) );

                % display method depends on number of array dimensions
                if ismatrix( x_dB )
                    imagesc( fftshift( x_dB.', 2 ), [ -dynamic_range_dB, 0 ] );
                    colorbar;
                else

                    %
                    warning( 'display_coefficients_vector:NotImplemented', 'Method display_coefficients_vector must be implemented!' );

                end % if ismatrix( x_dB )

            end % if LT.N_dimensions == 1

            % create title
            title( 'Fourier coefficients' );

        end % function display_coefficients_vector( LT, x, dynamic_range_dB, factor_dB )

	end % methods (Access = protected, Hidden)

end % classdef fourier < linear_transforms.linear_transform_vector
