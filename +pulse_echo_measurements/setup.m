%
% superclass for all pulse-echo measurement setups
%
% the class summarizes all constant objects
%
% author: Martin F. Schiffner
% date: 2018-03-12
% modified: 2019-02-03
%
classdef setup < handle

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        xdc_array ( 1, 1 ) transducer_models.transducer_array = transducer_models.L14_5_38	% transducer array
        xdc_behavior ( :, 1 ) pulse_echo_measurements.transfer_behavior                     % electromechanical transfer behaviors of all channels
        FOV ( 1, 1 ) fields_of_view.field_of_view = fields_of_view.orthotope( [4e-2, 4e-2], [-2e-2, 0.1] )	% field of view
        absorption_model ( 1, 1 ) absorption_models.absorption_model = absorption_models.time_causal( 0, 0.5, 1, 1540, 4e6, 1 )	% absorption model for the lossy homogeneous fluid
        c_avg = 1500;                                                                       % average small-signal sound speed
        f_clk = 80e6;                                                                       % frequency of the clock signal (Hz)
        str_name                                                                            % name of scan configuration

        % dependent properties
        D_ctr                   % mutual distances for all array elements
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = setup( xdc_array, transfer_behaviors, FOV, absorption_model, str_name )

            % internal properties
            object.xdc_array = xdc_array;
            object.xdc_behavior = transfer_behaviors;
            object.FOV = FOV;
            object.absorption_model = absorption_model;
            % assertion: independent properties form valid scan configuration

            object.str_name = str_name;

        end % function object = setup( xdc_array, FOV, absorption_model, str_name )

        %------------------------------------------------------------------
        % discretize pulse-echo measurement setup
        %------------------------------------------------------------------
        function discretize( object, N_interp_axis, delta_axis )

            %--------------------------------------------------------------
            % 1.) discretize transducer array and field of view
            %--------------------------------------------------------------
            object.xdc_array = discretize( object.xdc_array, N_interp_axis );
            object.FOV = discretize( object.FOV, delta_axis );

            %--------------------------------------------------------------
            % 2.) compute mutual distances for array elements
            %--------------------------------------------------------------
            % center coordinates (required for windowing RF data)
            object.D_ctr = sqrt( ( repmat( object.xdc_array.grid_ctr.positions( :, 1 ), [1, object.FOV.grid.N_points] ) - repmat( object.FOV.grid.positions( :, 1 )', [object.xdc_array.N_elements, 1] ) ).^2 + repmat( object.FOV.grid.positions( :, 2 )', [object.xdc_array.N_elements, 1] ).^2 );

        end % function discretize( object, N_interp_axis, delta_axis )

        %------------------------------------------------------------------
        % compute hash value
        %------------------------------------------------------------------
        function str_hash = hash( object )

            % use DataHash function to compute hash value
            str_hash = auxiliary.DataHash( object );

        end % function str_hash = hash( object )

    end % methods

end % classdef setup
