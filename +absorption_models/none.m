%
% none (lossless) absorption model
% author: Martin Schiffner
% date: 2016-08-25
%
classdef none < absorption_models.absorption_model

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        c_0             % constant phase and group velocity (m/s)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function ABS_none = none( c_0 )

            % create name string
            str_name = 'none';

            % constructor of superclass
            ABS_none@absorption_models.absorption_model( str_name );

            % internal properties
            ABS_none.c_0 = c_0;
        end

        %------------------------------------------------------------------
        % overload method: compute complex-valued wavenumbers
        %------------------------------------------------------------------
        function axis_k_tilde = compute_wavenumbers( ABS_none, axis_f )

            % compute real-valued wavenumbers using constant phase velocity
            axis_omega      = 2 * pi * axis_f;
            axis_k_tilde	= axis_omega / ABS_none.c_0;
        end

    end % methods

end % classdef none < absorption_models.absorption_model