%
% superclass for all symmetric pulse-echo measurement setups
%
% the class summarizes all constant objects
%
% author: Martin F. Schiffner
% date: 2019-01-24
% modified: 2019-01-24
%
classdef setup_symmetric < pulse_echo_measurements.setup

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % dependent properties
        D_ref                   % reference distances for first array element
        e_r_minus_r_s_ref_x     % reference unit vectors for first array element
        e_r_minus_r_s_ref_z     % reference unit vectors for first array element
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = setup_symmetric( xdc_array, FOV, absorption_model, str_name )

            %--------------------------------------------------------------
            % 1.) check symmetry of FOV about the axial axis
            %--------------------------------------------------------------
            must_equal_zero = FOV.offset_axis( 1:(FOV.N_dimensions - 1) ) + 0.5 * FOV.size_axis( 1:(FOV.N_dimensions - 1) );
            if ~all( abs( must_equal_zero ) < eps )
                errorStruct.message     = 'Symmetric pulse-echo measurement setup requires the symmetry of FOV about the axial axis!';
                errorStruct.identifier	= 'setup_symmetric:NoSymmetry';
                error( errorStruct );
            end
            % assertion: must_equal_zero is zero

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            object@pulse_echo_measurements.setup( xdc_array, FOV, absorption_model, str_name )
            % assertion: independent properties form valid scan configuration

        end % function object = setup_symmetric( xdc_array, FOV, absorption_model, str_name )

        %------------------------------------------------------------------
        % discretize pulse-echo measurement setup (overload discretize function)
        %------------------------------------------------------------------
        function discretize( object, N_interp_axis, delta_axis )

            %--------------------------------------------------------------
            % 1.) lateral spacing is an integer fraction of the element pitch
            %     => translational invariance by shifts of factor_interp_tx points
            %--------------------------------------------------------------
            must_equal_integer = object.xdc_array.element_pitch_axis( 1:(object.FOV.N_dimensions - 1) ) ./ delta_axis( 1:(object.FOV.N_dimensions - 1) );
            if ~all( must_equal_integer(:) == floor( must_equal_integer(:) ) )
                errorStruct.message     = 'Symmetric pulse-echo measurement setup requires the lateral spacings of the grid points in the FOV to be an integer fraction of the element pitch!';
                errorStruct.identifier	= 'discretize:NoIntegerFraction';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) discretize transducer array and field of view
            %--------------------------------------------------------------
            discretize@pulse_echo_measurements.setup( object, N_interp_axis, delta_axis );

            %--------------------------------------------------------------
            % 3.) compute mutual distances for array elements
            %--------------------------------------------------------------
            % compute reference distances for first array element (required for prefactors and scattering terms; N_interp_rx x FOV_N_points)
            object.D_ref = sqrt( ( repmat( object.xdc_array.grid( 1 ).positions( :, 1 ), [1, object.FOV.grid.N_points] ) - repmat( object.FOV.grid.positions( :, 1 )', [N_interp_axis(1), 1] ) ).^2 + repmat( object.FOV.grid.positions( :, 2 )', [N_interp_axis(1), 1] ).^2 );

            % compute reference unit vectors for first array element
            object.e_r_minus_r_s_ref_x = ( repmat( object.FOV.grid.positions( :, 1 )', [N_interp_axis(1), 1] ) - repmat( object.xdc_array.grid( 1 ).positions( :, 1 ), [1, object.FOV.grid.N_points] ) ) ./ object.D_ref;
            object.e_r_minus_r_s_ref_z = repmat( object.FOV.grid.positions( :, 2 )', [N_interp_axis(1), 1] ) ./ object.D_ref;

        end % function discretize( object, N_interp_axis, delta_axis )

    end % methods

end % classdef setup_symmetric
