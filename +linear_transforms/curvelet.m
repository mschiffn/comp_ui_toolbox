%
% superclass for all two- and three-dimensional discrete curvelet transforms
% (requires CurveLab-2.1.3: http://www-stat.stanford.edu/~wavelab)
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2020-11-11
%
classdef curvelet < linear_transforms.linear_transform_vector

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_points_axis ( 1, : ) double { mustBePositive, mustBeInteger, mustBeNonempty } = [ 512, 512 ]

        % dependent properties
        N_dimensions ( 1, 1 ) double { mustBeMember( N_dimensions, [ 2, 3 ] ), mustBeNonempty } = 2
        N_scales
        N_angles_scale
        N_coefficients_scale
        N_coefficients_angle_scale
        sizes

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = curvelet( N_points_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 1, 1 );

            % ensure cell array for N_points_axis
            if ~iscell( N_points_axis )
                N_points_axis = { N_points_axis };
            end

            % property validation function ensures valid N_points_axis

            %--------------------------------------------------------------
            % 2.) create discrete curvelet transforms
            %--------------------------------------------------------------
            % compute numbers of points
            N_points = cellfun( @( x ) prod( x( : ) ), N_points_axis );

            % maximum scale indices (fines scale) % nbscales = floor(log2(min(m,n)))-3;
            indices_scale_max = cellfun( @( x ) ceil( log2( min( x( : ) ) ) ), N_points_axis );

            % numbers of scales to be investigated (including coarsest scale)
            N_scales = indices_scale_max - 3;

            % number of angles at the 2nd coarsest level
            N_angles_2nd_coarse = 16;

            % determine numbers of coefficients and sizes
            N_angles_scale = cell( size( N_points_axis ) );
            N_coefficients = zeros( size( N_points_axis ) );
            N_coefficients_scale = cell( size( N_points_axis ) );
            N_coefficients_angle_scale = cell( size( N_points_axis ) );
            C = cell( size( N_points_axis ) );

            % iterate transforms
            for index_transform = 1:numel( N_points_axis )

                % number of coefficients for each angle on each scale
                N_coefficients_angle_scale{ index_transform } = cell( 1, N_scales( index_transform ) );

                % number of angles on each scale
                N_angles_scale{ index_transform } = [ 1, N_angles_2nd_coarse * 2.^( ceil( ( 0:( N_scales( index_transform ) - 2 ) ) / 2 ) ) ];
                N_angles_scale{ index_transform }( N_scales( index_transform ) ) = 1;                

                % dummy forward curvelet transform
                C{ index_transform } = fdct_wrapping( zeros( N_points_axis{ index_transform } ) );

                % coarsest scale
                N_coefficients_angle_scale{ index_transform }{ 1 } = numel( C{ index_transform }{ 1, 1 }{ 1, 1 } );
                C{ index_transform }{ 1, 1 }{ 1, 1 } = size( C{ index_transform }{ 1, 1 }{ 1, 1 } );

                % iterate intermediate scales
                for index_scale = 2:( N_scales( index_transform ) - 1 )

                    % initialize w/ zeros
                    N_coefficients_angle_scale{ index_transform }{ index_scale } = zeros( 1, N_angles_scale{ index_transform }( index_scale ) );

                    % iterate directions
                    for index_angle = 1:N_angles_scale{ index_transform }( index_scale )
                        N_coefficients_angle_scale{ index_transform }{ index_scale }( index_angle ) = numel( C{ index_transform }{ 1, index_scale }{ 1, index_angle } );
                        C{ index_transform }{ 1, index_scale }{ 1, index_angle } = size( C{ index_transform }{ 1, index_scale }{ 1, index_angle } );
                    end

                end % for index_scale = 2:( N_scales( index_transform ) - 1 )

                % finest scale
                N_coefficients_angle_scale{ index_transform }{ N_scales( index_transform ) } = numel( C{ index_transform }{ 1, N_scales( index_transform ) }{ 1, 1 } );
                C{ index_transform }{ 1, N_scales( index_transform ) }{ 1, 1 } = size( C{ index_transform }{ 1, N_scales( index_transform ) }{ 1, 1 } );

                % number of coefficients per scale
                N_coefficients_scale{ index_transform } = cellfun( @sum, N_coefficients_angle_scale{ index_transform } );

                % number of coefficients
                N_coefficients( index_transform ) = sum( N_coefficients_scale{ index_transform } );

            end % for index_transform = 1:numel( N_points_axis )

            % constructor of superclass
            objects@linear_transforms.linear_transform_vector( N_coefficients, N_points );

            % iterate discrete curvelet transforms
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).N_points_axis = N_points_axis{ index_object };

                % set dependent properties
                objects( index_object ).N_dimensions = numel( objects( index_object ).N_points_axis );
                objects( index_object ).N_scales = N_scales( index_object );
                objects( index_object ).N_angles_scale = N_angles_scale{ index_object };
                objects( index_object ).N_coefficients_scale = N_coefficients_scale{ index_object };
                objects( index_object ).N_coefficients_angle_scale = N_coefficients_angle_scale{ index_object };
                objects( index_object ).sizes = C{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = curvelet( N_points_axis )

        %------------------------------------------------------------------
        % format coefficients
        %------------------------------------------------------------------
        function y = format_coefficients( LTs, y )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class linear_transforms.curvelet
            if ~isa( LTs, 'linear_transforms.curvelet' )
                errorStruct.message = 'LTs must be linear_transforms.curvelet!';
                errorStruct.identifier = 'format_coefficients:NoCurveletTransforms';
                error( errorStruct );
            end

            % ensure cell array
            if ~iscell( y )
                y = { y };
            end

            % ensure equal number of dimensions and sizes
            [ LTs, y ] = auxiliary.ensureEqualSize( LTs, y );

            %--------------------------------------------------------------
            % 2.) format coefficients
            %--------------------------------------------------------------
            % iterate discrete curvelet transforms
            for index_object = 1:numel( LTs )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure numeric matrix
                if ~( isnumeric( y{ index_object } ) && ismatrix( y{ index_object } ) )
                    errorStruct.message = sprintf( 'y{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'format_coefficients:NoNumericMatrix';
                    error( errorStruct );
                end

                % ensure equal numbers of coefficients
                if size( y{ index_object }, 1 ) ~= LTs( index_object ).N_coefficients
                    errorStruct.message = sprintf( 'y{ %d } must have %d rows!', index_object, LTs( index_object ).N_coefficients );
                    errorStruct.identifier = 'format_coefficients:InvalidNumberOfRows';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) format coefficients for each transform
                %----------------------------------------------------------
                % partition coefficients into scales
                y{ index_object } = mat2cell( y{ index_object }, LTs( index_object ).N_coefficients_scale, size( y{ index_object }, 2 ) )';

                % coarsest scale
                y{ index_object }{ 1 } = { reshape( y{ index_object }{ 1 }, LTs( index_object ).sizes{ 1 }{ 1 } ) };

                % iterate intermediate scales
                for index_scale = 2:( LTs( index_object ).N_scales - 1 )

                    % partition coefficients on current scale into angles
                    y{ index_object }{ index_scale } = mat2cell( y{ index_object }{ index_scale }, LTs( index_object ).N_coefficients_angle_scale{ index_scale }, size( y{ index_object }{ index_scale }, 2 ) )';

                    % iterate directions
                    for index_angle = 1:LTs( index_object ).N_angles_scale( index_scale )

                        % iterate columns
%                         for index_vector = 1:size( y{ index_object }{ index_scale }{ index_angle }, 2 )

                            % reshape coefficients
                            y{ index_object }{ index_scale }{ index_angle } = reshape( y{ index_object }{ index_scale }{ index_angle }, LTs( index_object ).sizes{ index_scale }{ index_angle } );

%                         end % for index_vector = 1:size( y{ index_object }{ index_scale }{ index_angle }, 2 )

                    end % for index_angle = 1:LTs( index_object ).N_angles_scale( index_scale )

                end % for index_scale = 2:( LTs( index_object ).N_scales - 1 )

                % finest scale
                y{ index_object }{ LTs( index_object ).N_scales } = { reshape( y{ index_object }{ LTs( index_object ).N_scales }, LTs( index_object ).sizes{ LTs( index_object ).N_scales }{ 1 } ) };

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = format_coefficients( LTs, y )

        %------------------------------------------------------------------
        % vectorize coefficients
        %------------------------------------------------------------------
        function y = vectorize_coefficients( LT, C )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class linear_transforms.curvelet (scalar)
            if ~( isa( LT, 'linear_transforms.curvelet' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.curvelet!';
                errorStruct.identifier = 'vectorize_coefficients:NoSingleCurveletTransform';
                error( errorStruct );
            end

            % ensure cell array for C
            if ~( iscell( C ) && numel( C ) == LT.N_scales )
                errorStruct.message = 'C must be a cell array!';
                errorStruct.identifier = 'vectorize_coefficients:NoCellArray';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) vectorize coefficients
            %--------------------------------------------------------------
            % coarsest scale
            y = C{ 1, 1 }{ 1, 1 }( : );

            % iterate intermediate scales
            for index_scale = 2:( LT.N_scales - 1 )

                % iterate directions
                for index_angle = 1:LT.N_angles_scale( index_scale )
                    y = cat( 1, y, C{ 1, index_scale }{ 1, index_angle }( : ) );
                end

            end % for index_scale = 2:( LT.N_scales - 1 )

            % finest scale
            y = cat( 1, y, C{ 1, LT.N_scales }{ 1, 1 }( : ) );

        end % function y = vectorize_coefficients( LT, C )

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
            % ensure class linear_transforms.curvelet (scalar)
            if ~( isa( LT, 'linear_transforms.curvelet' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.curvelet!';
                errorStruct.identifier = 'forward_transform_vector:NoSingleCurveletTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward curvelet transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            % apply forward transform
            C = fdct_wrapping( x );

            % coarsest scale
            y = vectorize_coefficients( LT, C );

        end % function y = forward_transform_vector( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single vector)
        %------------------------------------------------------------------
        function y = adjoint_transform_vector( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.curvelet (scalar)
            if ~( isa( LT, 'linear_transforms.curvelet' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.curvelet!';
                errorStruct.identifier = 'adjoint_transform_vector:NoSingleFourierTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint curvelet transform (single vector)
            %--------------------------------------------------------------
            % format coefficients
            C = format_coefficients( LT, x );

            % apply adjoint transform
            y = ifdct_wrapping( C );

            % return result as column vector
            y = y( : );

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

            % ensure class linear_transforms.curvelet (scalar)
            if ~( isa( LT, 'linear_transforms.curvelet' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be a single linear_transforms.curvelet!';
                errorStruct.identifier = 'display_coefficients_vector:NoSingleCurveletTransform';
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
            % format coefficients
            C = format_coefficients( LT, x );

            % check dimensionality of the discrete Fourier transform
            if LT.N_dimensions == 2

                %----------------------------------------------------------
                % a) two-dimensional discrete curvelet transform
                %----------------------------------------------------------
                % convert to image
                img = fdct_wrapping_dispcoef( C );

                % logarithmic compression
                img_dB = illustration.dB( img, factor_dB );

                % display method depends on number of array dimensions
                imagesc( img_dB, [ -dynamic_range_dB, 0 ] );
                colorbar;
                colormap gray;

            else

                %----------------------------------------------------------
                % b) three-dimensional discrete curvelet transform
                %----------------------------------------------------------
                %
                warning( 'display_coefficients_vector:NotImplemented', 'Method display_coefficients_vector must be implemented!' );

            end % if LT.N_dimensions == 2

            % create title
            title( 'Curvelet coefficients' );

        end % function display_coefficients_vector( LT, x, dynamic_range_dB, factor_dB )

	end % methods (Access = protected, Hidden)

end % classdef curvelet < linear_transforms.linear_transform_vector
