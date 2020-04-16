%
% one-, two-, or three-dimensional
% discrete symmetric wave atom transforms for
% various options
%
% requires: WaveAtom toolbox in Matlab v.1.1.1 (April 2008)
%           (http://www.waveatom.org/software.html)
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2020-04-16
%
classdef wave_atom < linear_transforms.linear_transform_vector

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        type ( 1, 1 ) linear_transforms.wave_atoms.type { mustBeNonempty } = linear_transforms.wave_atoms.orthogonal
        N_dimensions ( 1, 1 ) { mustBePositive, mustBeInteger, mustBeNonempty } = 2         % number of dimensions
        scale_finest ( 1, 1 ) { mustBeNonnegative, mustBeInteger, mustBeNonempty } = 9      % finest scale ( fine level )

        % dependent properties
        N_points_axis ( 1, : ) double { mustBePositive, mustBeInteger, mustBeNonempty } = [ 512, 512 ]
        handle_fwd ( 1, 1 ) function_handle { mustBeNonempty } = @( x ) x	% function handle to forward transform
        handle_inv ( 1, 1 ) function_handle { mustBeNonempty } = @( x ) x	% function handle to inverse transform

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = wave_atom( types, N_dimensions, scales_finest )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure string array for types
            if ~isa( types, 'linear_transforms.wave_atoms.type' )
                errorStruct.message = 'types must be linear_transforms.wave_atoms.type!';
                errorStruct.identifier = 'wave_atom:NoWaveAtomTypes';
                error( errorStruct );
            end

            % property validation function ensures nonempty positive integers for N_dimensions

            % property validation function ensures nonempty positive integers for scales_finest

            % ensure equal number of dimensions and sizes
            [ types, N_dimensions, scales_finest ] = auxiliary.ensureEqualSize( types, N_dimensions, scales_finest );

            %--------------------------------------------------------------
            % 2.) create discrete wave atom transforms
            %--------------------------------------------------------------
            % compute numbers of grid points
            N_points = ( 2.^scales_finest ).^N_dimensions;

            % extract numbers of layers
            N_layers = reshape( [ types.N_layers ], size( types ) );

            % compute numbers of transform coefficients
            N_coefficients = N_layers .* N_points;

            % constructor of superclass
            objects@linear_transforms.linear_transform_vector( N_coefficients, N_points );

            % iterate discrete wave atom transforms
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).type = types( index_object );
                objects( index_object ).N_dimensions = N_dimensions( index_object );
                objects( index_object ).scale_finest = scales_finest( index_object );

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                % number of points along each axis
                objects( index_object ).N_points_axis = repmat( 2.^objects( index_object ).scale_finest, [ 1, objects( index_object ).N_dimensions ] );

                % specify transform functions
                switch objects( index_object ).N_dimensions

                    case 1

                        %--------------------------------------------------
                        % i.) one-dimensional transform
                        %--------------------------------------------------
                        objects( index_object ).handle_fwd = @fwa1sym;
                        objects( index_object ).handle_inv = @iwa1sym;

                    case 2

                        %--------------------------------------------------
                        % ii.) two-dimensional transform
                        %--------------------------------------------------
                        objects( index_object ).handle_fwd = @fwa2sym;
                        objects( index_object ).handle_inv = @iwa2sym;

                    case 3

                        %--------------------------------------------------
                        % iii.) three-dimensional transform
                        %--------------------------------------------------
                        objects( index_object ).handle_fwd = @fwa3sym;
                        objects( index_object ).handle_inv = @iwa3sym;

                    otherwise

                        %--------------------------------------------------
                        % iv.) invalid number of dimensions
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'objects( %d ).N_dimensions must equal 1 or 2!', index_object );
                        errorStruct.identifier = 'wave_atom:InvalidNumberDimensions';
                        error( errorStruct );

                end % switch objects( index_object ).N_dimensions

            end % for index_object = 1:numel( objects )

        end % function objects = wave_atom( types, N_dimensions, scales_finest )

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
            % calling function ensures class linear_transforms.linear_transform_vector (scalar) for LT
            % calling function ensures numeric column vector for x
            % calling function ensures equal numbers of points

            % ensure class linear_transforms.wave_atom (scalar)
            if ~( isa( LT, 'linear_transforms.wave_atom' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.wave_atom!';
                errorStruct.identifier = 'forward_transform_vector:NoSingleWaveAtomTransform';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute forward wave atom transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            % apply forward transform
            switch class( LT.type )
                case 'linear_transforms.wave_atoms.orthogonal'
                    y_act = LT.handle_fwd( x, LT.type.pat, 'ortho' );
                case 'linear_transforms.wave_atoms.directional'
                    y_act = LT.handle_fwd( x, LT.type.pat, 'directional' );
                case 'linear_transforms.wave_atoms.complex'
                    y_act = LT.handle_fwd( x, LT.type.pat, 'complex' );
            end

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
            % calling function ensures class linear_transforms.linear_transform_vector (scalar) for LT
            % calling function ensures numeric column vector for x
            % calling function ensures equal numbers of coefficients

            % ensure class linear_transforms.wave_atom (scalar)
            if ~( isa( LT, 'linear_transforms.wave_atom' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.wave_atom!';
                errorStruct.identifier = 'adjoint_transform_vector:NoSingleWaveAtomTransform';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute adjoint wave atom transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            x_act = reshape( x, [ LT.N_points_axis, LT.type.N_layers ] );

            % apply adjoint transform
            switch class( LT.type )
                case 'linear_transforms.wave_atoms.orthogonal'
                    y_act = LT.handle_inv( x_act, LT.type.pat, 'ortho' );
                case 'linear_transforms.wave_atoms.directional'
                    y_act = LT.handle_inv( x_act, LT.type.pat, 'directional' );
                case 'linear_transforms.wave_atoms.complex'
                    y_act = LT.handle_inv( x_act, LT.type.pat, 'complex' );
            end

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

            % ensure class linear_transforms.wave_atom (scalar)
            if ~( isa( LT, 'linear_transforms.wave_atom' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.wave_atom!';
                errorStruct.identifier = 'display_coefficients_vector:NoSingleWaveAtomTransform';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) display coefficients (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            x_act = reshape( x, [ LT.N_points_axis, LT.type.N_layers ] );

            % logarithmic compression
            x_act_dB = illustration.dB( x_act, 10 );

            % display vector
            switch LT.type.N_layers
                case 1
                    if LT.N_dimensions == 2
                        imagesc( x_act_dB, [ -60, 0 ] );
                    end
                case 2
                    if LT.N_dimensions == 2
                        subplot( 1, 2, 1 );
                        imagesc( x_act_dB( :, :, 1 )', [ -60, 0 ] );
                        subplot( 1, 2, 2 );
                        imagesc( x_act_dB( :, :, 2 )', [ -60, 0 ] );
                    end
                case 4
                    if LT.N_dimensions == 2
                        subplot( 2, 2, 1 );
                        imagesc( x_act_dB( :, :, 1 )', [ -60, 0 ] );
                        title('Layer 1');
                        subplot( 2, 2, 2 );
                        imagesc( x_act_dB( :, :, 2 )', [ -60, 0 ] );
                        title('Layer 2');
                        subplot( 2, 2, 3 );
                        imagesc( x_act_dB( :, :, 3 )', [ -60, 0 ] );
                        title('Layer 3');
                        subplot( 2, 2, 4 );
                        imagesc( x_act_dB( :, :, 4 )', [ -60, 0 ] );
                        title('Layer 4');
                    end
            end

        end % function display_coefficients_vector( LT, x )

	end % methods (Access = protected, Hidden)

end % classdef wave_atom < linear_transforms.linear_transform_vector
