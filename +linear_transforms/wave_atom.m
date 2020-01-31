%
% superclass for all discrete wave atom transforms
%
% requires: WaveAtoms (http://www-stat.stanford.edu/~wavelab)
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2020-01-30
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
            auxiliary.mustBeEqualSize( types, N_dimensions, scales_finest );

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
                        errorStruct.identifier = 'wavelet:InvalidNumberDimensions';
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
            % ensure class linear_transforms.wave_atom
            if ~( isa( LT, 'linear_transforms.wave_atom' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.wave_atom!';
                errorStruct.identifier = 'forward_transform_single:NoSingleWaveAtomTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward wave atom transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            % apply forward transform
            y_act = LT.handle_fwd( x, 'p', 'ortho' );

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
            % ensure class linear_transforms.wave_atom
            if ~( isa( LT, 'linear_transforms.wave_atom' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.wave_atom!';
                errorStruct.identifier = 'adjoint_transform_single:NoSingleWaveAtomTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint wave atom transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            x_act = reshape( x, [ LT.N_points_axis, LT.type.N_layers ] );

            % apply adjoint transform
            y_act = LT.handle_inv( x_act, 'p', 'ortho' );

            % return result as column vector
            y = y_act( : );

        end % function y = adjoint_transform_vector( LT, x )

	end % methods (Access = protected, Hidden)

end % classdef wave_atom < linear_transforms.linear_transform_vector
