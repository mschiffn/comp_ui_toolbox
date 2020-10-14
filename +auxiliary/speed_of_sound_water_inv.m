function T = speed_of_sound_water_inv( c_water )
%
% Computes the temperature of pure water for specified speeds of sound
%
% The function inverts a fifth-degree polynomial least-squares fit to
% 147 observations between 0.001 °C and 95.1264 °C on the T_{68} scale
% (see [1]) using the MATLAB Optimization Toolbox.
%
% INPUT:
%   c_water = array of speeds of sound in pure water (m/s)
%
% OUTPUT:
%   T = array of temperatures (°C, T_{68} scale)
%
% REFERENCES:
%	[1] V. A. Del Grosso, C. W. Mader, "Speed of sound in pure water,"
%       J. Acoust. Soc. Am., vol. 52, no. 5B, pp. 1442–1446, Nov. 1972.
%       DOI: 10.1121/1.1913258
%
% REMARKS:
%	- Ref. [1] claims to use 148 observations, but listed only 147
%	  (cf. data below).
%	- The polynomial coefficients in [1, Table III] could not be validated.
%	- The polynomial coefficients reproduce [1, Table IV] except for very high temperatures.
%
% author: Martin F. Schiffner
% data: 2020-01-13
% modified: 2020-09-10

    %----------------------------------------------------------------------
    % 1.) check arguments
    %----------------------------------------------------------------------
    % ensure one argument
    narginchk( 1, 1 );

	% ensure class physical_values.meter_per_second
	if ~isa( c_water, 'physical_values.meter_per_second' )
        errorStruct.message = 'c_water must be physical_values.meter_per_second!';
        errorStruct.identifier = 'speed_of_sound_water_inv:NoMetersPerSecond';
        error( errorStruct );
    end
    c_water = double( c_water );

	% ensure valid speeds of sound
	indicator = c_water < 1402.395 || c_water > 1555.145;
	if any( indicator( : ) )
        warning( 'At least one speed of sound is outside the observed range!' );
	end

	%----------------------------------------------------------------------
	% 2.) coefficients of fifth-degree polynomial
	%----------------------------------------------------------------------
    coef = [ 0.140238744511433878869866020977497100830078125e4, ...
             0.503715890656311326978311626589857041835784912109375e1, ...
            -0.58090835079566895127189951608670526184141635894775390625e-1, ...
             0.334402350380210116979895484945473072002641856670379638671875e-3, ...
            -0.14807647561498290376025450953978435109092970378696918487548828125e-5, ...
             0.3158928530215746731866020334078375142450312296205083839595317840576171875e-8 ];

    %----------------------------------------------------------------------
    % 3.) invert polynomial
    %----------------------------------------------------------------------
	% initial state
    T_0 = 20;

	% boundaries
	T_lbs = 0;
	T_ubs = 100;

	% set optimization options
	options_optimization = optimoptions( 'fsolve', 'Algorithm', 'trust-region-reflective', 'FunValCheck', 'on', 'Diagnostics', 'on', 'Display', 'iter-detailed', 'FunctionTolerance', 1e-10, 'OptimalityTolerance', 1e-10, 'StepTolerance', 1e-10, 'SpecifyObjectiveGradient', true, 'CheckGradients', false, 'FiniteDifferenceType', 'central', 'FiniteDifferenceStepSize', 1e-10, 'MaxFunctionEvaluations', 5e3, 'MaxIterations', 5e3 );

    % find solutions to nonlinear equation
    [ T, F, exitflag, output, JAC ] = fsolve( @C, T_0, options_optimization );

    % assign physical unit
%     T = physical_values.degree_celsius( T );

    %----------------------------------------------------------------------
    % nested objective function
    %----------------------------------------------------------------------
	function [ y, J ] = C( T )

        % compute speed of sound minus c_water
        y = coef( 1 ) - c_water + coef( 2 ) * T + coef( 3 ) * T.^2 + coef( 4 ) * T.^3 + coef( 5 ) * T.^4 + coef( 6 ) * T.^5;

        % check if Jacobian is required
        if nargout > 1

            % compute Jacobian
            J = coef( 2 ) + 2 * coef( 3 ) * T + 3 * coef( 4 ) * T.^2 + 4 * coef( 5 ) * T.^3 + 5 * coef( 6 ) * T.^4;

        end % if nargout > 1

    end % function [ y, J ] = C( theta )

end % function c_water = speed_of_sound_water_inv( T )
