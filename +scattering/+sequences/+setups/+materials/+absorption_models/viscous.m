%
% viscous absorption model
% author: Martin Schiffner
% date: 2016-08-25
%
classdef viscous < absorption_models.absorption_model

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        viscosity_kinematic     % kinematic viscosity (Pa s)
        c_0                     % phase velocity (m / s)
        relaxation_time         % relaxation time (s)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function ABS_viscous = viscous( viscosity_kinematic, c_0 )

            % create name string
            str_name = sprintf( 'viscous_%.2f', viscosity_kinematic );

            % constructor of superclass
            ABS_viscous@absorption_models.absorption_model( str_name );

            % internal properties
            ABS_viscous.viscosity_kinematic	= viscosity_kinematic;
            ABS_viscous.c_0                 = c_0;
            ABS_viscous.relaxation_time     = viscosity_kinematic / c_0^2;
        end

        %------------------------------------------------------------------
        % overload method: compute complex-valued wavenumbers
        %------------------------------------------------------------------
        function axis_k_tilde = compute_wavenumbers( ABS_viscous, axis_f )

            % compute real-valued wavenumbers using limiting phase velocity
            axis_omega	= 2 * pi * axis_f;
            axis_k_0	= axis_omega / ABS_viscous.c_0;

            % compute complex-valued wavenumbers
            axis_k_tilde_squared = axis_k_0.^2 ./ ( 1 + 1j * axis_omega * ABS_viscous.relaxation_time );
            axis_k_tilde = sqrt( axis_k_tilde_squared );

        end

    end % methods

end % classdef viscous < absorption_models.absorption_model
