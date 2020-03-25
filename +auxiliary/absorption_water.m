function alpha = absorption_water( T )
%
% Computes the power-law absorption coefficient in pure water at specified temperatures
%
% The function uses a seventh-degree polynomial least-squares fit to
% 9 measurements (see [1]).
%
% INPUT:
%   T = array of temperatures (°C)
%
% OUTPUT:
%   alpha = power-law absorption coefficient in pure water (dB / MHz^2 / cm)
%
% REFERENCES:
%   [1] J. M. M. Pinkerton, "The absorption of ultrasonic waves in liquids and its relation to molecular constitution,"
%       P. Phys. Soc. B, vol. 62, no. 2, pp. 129-141, Feb. 1949
%       DOI: 10.1088/0370-1301/62/2/307
%
% author: Martin F. Schiffner
% data: 2020-01-13
% modified: 2020-03-21

    %----------------------------------------------------------------------
    % 1.) check arguments
    %----------------------------------------------------------------------
    % ensure class physical_values.degree_celsius
%     if ~isa( T, 'physical_values.degree_celsius' )
%         errorStruct.message = 'T must be physical_values.degree_celsius!';
%         errorStruct.identifier = 'absorption_water:NoDegreesCelsius';
%         error( errorStruct );
%     end

    %----------------------------------------------------------------------
    % 2.) measurement data for least-squares estimate (see [1, Table 1])
	%----------------------------------------------------------------------
	% Table 1. Absorption of Sound in Water
	% Measured values compared with those predicted from Stokes' formula
	% T ( °C ) | \alpha / \nu^{2} ( 1e-17 s^{2} / cm ) | probable error (%)
% 	data = [ ...
%          0, 56.9,  0.6; ...
%          5, 44.1,  0.6; ...
%         10, 35.8,  0.86; ...
%         15, 29.8,  0.9; ...
%         20, 25.3,  1.35; ...
%         30, 19.1,  1.7; ...
%         40, 14.61, 0.5; ...
%         50, 11.99, 0.4; ...
%         60, 10.15, 0.8 ];
% 
% 	% least-squares estimate
% 	N_degrees = 7;
% 	N_coef = N_degrees + 1;
% 	X = ones( size( data, 1 ), N_coef );
% 	for index_coef = 2:N_coef
%         X( :, index_coef ) = data( :, 1 ).^( index_coef - 1 );
%     end
% 	coef = X \ data( :, 2 );

	%----------------------------------------------------------------------
	% 3.) coefficients of seventh-degree polynomial
	%----------------------------------------------------------------------
	coef = [ 56.89977005377601670943477074615657329559326171875, ...
             -3.2828544967536803511620746576227247714996337890625, ...
              0.1848832025529568434674132504369481466710567474365234375, ...
             -0.009683714721873591668721559244659147225320339202880859375, ...
              0.0003669563650229056071642996794679447702947072684764862060546875, ...
             -0.000008487030816199488320906996197123817182728089392185211181640625, ...
              0.0000001043428773886832632731834252966252041261441263486631214618682861328125, ...
             -0.00000000051730976792878116144402554218094912596992429598685703240334987640380859375 ];

    %----------------------------------------------------------------------
    % 4.) evaluate polynomial
    %----------------------------------------------------------------------
    exponents = ( 0:( numel( coef ) - 1 ) );
    alpha = coef( 1 ) * ones( size( T ) );
    for index_summand = 2:numel( exponents )
        alpha = alpha + coef( index_summand ) * double( T ).^exponents( index_summand );
    end

    % assign physical unit
    %   \alpha / \nu^{2} = 29.8e-17 s^{2} / cm
	%	=> \alpha = 29.8e-5 * \nu^{2} / ( MHz^2 * cm )
    %
    % - logarithmic scale:
    %   20 * log10( exp( - \alpha x ) ) = 20 * ln( exp( - \alpha x ) ) / ln( 10 ) = - \alpha 20 / ln( 10 ) x
    %   => \alpha_{dB} = 29.8e-5 * 20 * \nu^{2} / ln( 10 ) / ( MHz^2 * cm ) = 0.002588395112143380798197522807413406553678214550018310546875 * \nu^{2} / ( MHz^2 * cm )
    %   => \alpha = 2.588395e-3 dB / ( MHz^2 * cm )
    % T = 16.3: 28.5087e-17 * 1e12 * 20 / log( 10 ) dB / ( MHz^2 * cm ) = 2.4762e-3 dB / ( MHz^2 * cm )
    alpha = alpha * 1e-5 * 20 / log( 10 );

end % function alpha = absorption_water( T )
