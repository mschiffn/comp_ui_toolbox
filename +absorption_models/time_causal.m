%
% time causal absorption model
% author: Martin Schiffner
% date: 2016-08-25
%
classdef time_causal < absorption_models.absorption_model

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        alpha_0                         % Np / m
        alpha_1                         % Np / (Hz^exponent * m)
        absorption_constant             % dB / cm
        absorption                      % dB / (MHz^exponent * cm)
        exponent                        % exponent in power law (1)
        c_ref                           % reference phase velocity (m/s)
        f_ref                           % temporal reference frequency for reference phase velocity c_ref (Hz)
        flag_dispersion = 1;            % include frequency-dependent dispersion (causal model) if nonzero
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function ABS_time_causal = time_causal( constant, slope, exponent, c_ref, f_ref, flag_dispersion )

            % create name string
            str_name = sprintf( 'power_law_%.2f_%.2f_%.2f', constant, slope, exponent );

            % constructor of superclass
            ABS_time_causal@absorption_models.absorption_model( str_name );

            % time causal model (Szabo 2004)
            ABS_time_causal.alpha_0 = constant * log(10) / (20 * 0.01);
            ABS_time_causal.alpha_1 = slope * log(10) / (20 * 0.01 * (1e6)^exponent);

            % internal properties
            ABS_time_causal.absorption_constant     = constant;
            ABS_time_causal.absorption              = slope;
            ABS_time_causal.exponent                = exponent;
            ABS_time_causal.c_ref                   = c_ref;
            ABS_time_causal.f_ref                   = f_ref;
            ABS_time_causal.flag_dispersion         = flag_dispersion;
        end

        %------------------------------------------------------------------
        % overload method: compute complex-valued wavenumbers
        %------------------------------------------------------------------
        function axis_k_tilde = compute_wavenumbers( ABS_time_causal, axis_f )

            % compute real-valued wavenumbers using reference phase velocity
            axis_k_ref = 2 * pi * axis_f / ABS_time_causal.c_ref;

            % compute imaginary part (even function of temporal frequency)
            axis_k_tilde_imag = - ABS_time_causal.alpha_0 - ABS_time_causal.alpha_1 * abs( axis_f ).^ABS_time_causal.exponent;

            % compute real part (odd function of temporal frequency)
            if ABS_time_causal.flag_dispersion

                % include frequency-dependent dispersion (causal model)
                if mod( ABS_time_causal.exponent, 2 ) == 0 || floor( ABS_time_causal.exponent ) ~= ABS_time_causal.exponent

                    % exponent is an even integer or noninteger
                    axis_k_tilde_real = axis_k_ref + ABS_time_causal.alpha_1 * tan( ABS_time_causal.exponent * pi / 2 ) * axis_f .* ( abs( axis_f ).^(ABS_time_causal.exponent - 1) - abs( ABS_time_causal.f_ref )^(ABS_time_causal.exponent - 1) );
                else

                    % exponent is an odd integer
                    axis_k_tilde_real = axis_k_ref - 2 * ABS_time_causal.alpha_1 * axis_f.^ABS_time_causal.exponent .* log( abs( axis_f / ABS_time_causal.f_ref ) ) / pi;
                end
            else

                % ignore frequency-dependent dispersion (noncausal model)
                axis_k_tilde_real = axis_k_ref;
            end

            % check for zero frequency and ensure odd function of temporal frequency
            indicator = abs( axis_f ) < eps;
            axis_k_tilde_real( indicator ) = 0;

            % compose complex-valued wavenumbers
            axis_k_tilde = axis_k_tilde_real + 1j * axis_k_tilde_imag;

        end

    end % methods

end % classdef time_causal < absorption_models.absorption_model
