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

        
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = setup_symmetric( xdc_array, FOV, absorption_model )

            

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            object@pulse_echo_measurements.setup( xdc_array, FOV, absorption_model )
            % assertion: independent properties form valid scan configuration

        end % function object = setup_symmetric( xdc_array, FOV, absorption_model, str_name )

        %------------------------------------------------------------------
        % discretize pulse-echo measurement setup (overload discretize function)
        %------------------------------------------------------------------
%         function discretize( object, N_interp_axis, delta_axis )
% 

% 
%             %--------------------------------------------------------------
%             % 2.) discretize transducer array and field of view
%             %--------------------------------------------------------------
%             discretize@pulse_echo_measurements.setup( object, N_interp_axis, delta_axis );
% 

%         end % function discretize( object, N_interp_axis, delta_axis )

    end % methods

end % classdef setup_symmetric
