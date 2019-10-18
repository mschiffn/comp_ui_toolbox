%
% single relaxation absorption model
% author: Martin Schiffner
% date: 2017-08-18
%
classdef relaxation_single < absorption_models.absorption_model

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        c_0                 % low-frequency limit of the phase velocity (m / s) [equilibrium]
        c_inf               % high-frequency limit of the phase velocity (m / s) [frozen]
        relaxation_time     % relaxation time (s)
        dispersion          % dispersion of the relaxation (1)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function ABS_relax_sing = relaxation_single( c_0, c_inf, relaxation_time )

            % create name string
            str_name = sprintf( 'relaxation_single_%.1f_%.1f_%.2f', c_0, c_inf, relaxation_time * 1e6 );

            % constructor of superclass
            ABS_relax_sing@absorption_models.absorption_model( str_name );

            % internal properties
            ABS_relax_sing.c_0              = c_0;
            ABS_relax_sing.c_inf            = c_inf;
            ABS_relax_sing.relaxation_time	= relaxation_time;
            ABS_relax_sing.dispersion       = ( c_inf / c_0 )^2 - 1;
        end

        %------------------------------------------------------------------
        % overload method: compute complex-valued wavenumbers
        %------------------------------------------------------------------
        function axis_k_tilde = compute_wavenumbers( ABS_relax_sing, axis_f )

            % compute real-valued wavenumbers using limiting phase velocity
            axis_omega	= 2 * pi * axis_f;
            axis_k_0	= axis_omega / ABS_relax_sing.c_0;

            % compute auxiliary values (\cite[(C-7) in][319]{book:Blackstock2000})
            summand     = 1j * axis_omega * ABS_relax_sing.relaxation_time;
            numerator	= 1 + summand;
            denominator = 1 + summand * ( ABS_relax_sing.dispersion + 1 );

            % compute complex-valued wavenumbers
            axis_k_tilde_norm_squared = numerator ./ denominator;
            axis_k_tilde = axis_k_0 .* sqrt( axis_k_tilde_norm_squared );
        end

    end % methods

end % classdef relaxation_single < absorption_models.absorption_model
