function c_water = speed_of_sound_water( T )
%
% Computes the speed of sound in pure water at specified temperatures
%
% The function uses a fifth-degree polynomial least-squares fit to
% 147 observations between 0.001 °C and 95.1264 °C on the T_{68} scale
% (see [1]).
%
% INPUT:
%   T = array of temperatures (°C, T_{68} scale)
%
% OUTPUT:
%   c_water = array of speeds of sound in pure water (m/s)
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
% modified: 2020-01-14

    %----------------------------------------------------------------------
    % 1.) check arguments
    %----------------------------------------------------------------------
    % ensure class physical_values.degree_celsius
    if ~isa( T, 'physical_values.degree_celsius' )
        errorStruct.message = 'T must be physical_values.degree_celsius!';
        errorStruct.identifier = 'speed_of_sound_water:NoDegreesCelsius';
        error( errorStruct );
    end

	% ensure valid temperatures
	indicator = T < 0.001 || T > 95.1264;
	if any( indicator( : ) )
        warning( 'At least one temperature is outside the observed range!' );
	end

    %----------------------------------------------------------------------
    % 2.) measurement data for least-squares estimate (see [1, Tables I, II])
    %----------------------------------------------------------------------
    % Table I (112 observations)
%     data_1 = [ ...
%     0.0010, 1402.395; ...
%     0.0020, 1402.398; ...
%     0.0030, 1402.404; ...
%     0.0030, 1402.406; ...
%     0.0110, 1402.445; ...
%     0.0120, 1402.448; ...
%     0.0130, 1402.456; ...
%     0.0130, 1402.453; ...
%     0.0140, 1402.459; ...
%     0.0520, 1402.649; ...
%     0.0520, 1402.652; ...
%     0.0520, 1402.649; ...
%     0.0530, 1402.654; ...
%     0.0530, 1402.654; ...
%     0.1979, 1403.383; ...
%     0.1979, 1403.383; ...
%     0.1989, 1403.390; ...
%     0.1989, 1403.388; ...
%     0.1989, 1403.388; ...
%     0.4878, 1404.829; ...
%     0.4898, 1404.843; ...
%     0.4908, 1404.848; ...
%     0.4988, 1404.888; ...
%     0.5008, 1404.894; ...
%     0.5018, 1404.901; ...
%     1.0005, 1407.365; ...
%     1.0025, 1407.377; ...
%     1.0025, 1407.382; ...
%     1.0035, 1407.384; ...
%     1.0035, 1407.384; ...
%     1.0045, 1407.392; ...
%     1.0045, 1407.386; ...
%     1.0055, 1407.391; ...
%     1.0095, 1407.412; ...
%     1.0175, 1407.451; ...
%     1.0235, 1407.482; ...
%     1.0305, 1407.516; ...
%     2.0490, 1412.468; ...
%     2.0560, 1412.501; ...
%     2.0620, 1412.527; ...
%     2.0650, 1412.543; ...
%     2.0680, 1412.554; ...
%     2.0720, 1412.574; ...
%     2.4868, 1414.553; ...
%     2.4868, 1414.556; ...
%     2.4898, 1414.573; ...
%     2.4918, 1414.582; ...
%     2.4928, 1414.585; ...
%     2.9736, 1416.861; ...
%     2.9746, 1416.864; ...
%     2.9766, 1416.875; ...
%     2.9766, 1416.876; ...
%     3.4913, 1419.279; ...
%     3.4913, 1419.277; ...
%     3.4923, 1419.277; ...
%     3.4923, 1419.280; ...
%     3.4933, 1419.287; ...
%     3.7972, 1420.702; ...
%     3.7982, 1420.694; ...
%     3.7992, 1420.700; ...
%     3.8002, 1420.707; ...
%     3.8002, 1420.707; ...
%     3.9911, 1421.584; ...
%     3.9911, 1421.587; ...
%     3.9921, 1421.590; ...
%     3.9921, 1421.589; ...
%     3.9931, 1421.595; ...
%     4.2160, 1422.620; ...
%     4.2170, 1422.624; ...
%     4.2170, 1422.622; ...
%     4.2170, 1422.622; ...
%     4.5269, 1424.032; ...
%     4.5279, 1424.039; ...
%     4.5279, 1424.040; ...
%     4.5279, 1424.039; ...
%     5.4935, 1428.364; ...
%     5.4935, 1428.365; ...
%     5.4945, 1428.367; ...
%     5.4965, 1428.378; ...
%     5.9892, 1430.543; ...
%     5.9902, 1430.548; ...
%     5.9902, 1430.551; ...
%     5.9922, 1430.559; ...
%     5.9952, 1430.572; ...
%     7.9894, 1439.089; ...
%     7.9904, 1439.094; ...
%     7.9904, 1439.096; ...
%     7.9904, 1439.094; ...
%     7.9914, 1439.102; ...
%     9.9537, 1447.087; ...
%     9.9537, 1447.087; ...
%     9.9547, 1447.094; ...
%     9.9547, 1447.091; ...
%     9.9547, 1447.089; ...
%     39.9657, 1528.809; ...
%     39.9777, 1528.831; ...
%     39.9887, 1528.847; ...
%     59.9924, 1550.980; ...
%     60.0034, 1550.986; ...
%     60.0124, 1550.994; ...
%     60.0204, 1550.998; ...
%     60.0294, 1551.004; ...
%     70.1190, 1554.819; ...
%     70.1210, 1554.819; ...
%     70.1240, 1554.819; ...
%     70.1340, 1554.824; ...
%     70.1500, 1554.824; ...
%     90.0858, 1550.430; ...
%     90.0868, 1550.430; ...
%     95.1214, 1547.096; ...
%     95.1224, 1547.100; ...
%     95.1264, 1547.095 ];

	% Table II (35 observations)
%     data_2 = [ ...
%     0.0560, 1402.673; ...
%     0.0610, 1402.695; ...
%     0.0640, 1402.705; ...
%     0.0680, 1402.726; ...
%     0.0720, 1402.747; ...
%     4.9887, 1426.115; ...
%     4.9917, 1426.126; ...
%     4.9927, 1426.129; ...
%     4.9937, 1426.131; ...
%     9.9917, 1447.234; ...
%     9.9957, 1447.249; ...
%     10.0027, 1447.276; ...
%     10.0117, 1447.307; ...
%     19.9196, 1482.091; ...
%     19.9206, 1482.096; ...
%     19.9216, 1482.102; ...
%     24.9815, 1496.636; ...
%     24.9855, 1496.646; ...
%     29.9816, 1509.081; ...
%     29.9836, 1509.089; ...
%     34.9710, 1519.752; ...
%     34.9810, 1519.768; ...
%     34.9870, 1519.781; ...
%     39.9727, 1528.820; ...
%     39.9747, 1528.823; ...
%     39.9777, 1528.827; ...
%     39.9847, 1528.837; ...
%     49.9956, 1542.545; ...
%     50.0126, 1542.563; ...
%     50.0366, 1542.591; ...
%     50.0466, 1542.602; ...
%     60.0194, 1550.999; ...
%     60.0124, 1550.999; ...
%     73.9957, 1555.144; ...
%     74.0117, 1555.144; ...
%     74.0218, 1555.145 ];

	% combine data
	% data = [ data_1; data_2 ];

	% least-squares estimate
% 	N_degrees = 5;
% 	N_coef = N_degrees + 1;
% 	X = ones( size( data, 1 ), N_coef );
% 	for index_coef = 2:N_coef
%       X( :, index_coef ) = data( :, 1 ).^( index_coef - 1 );
%   end
% 	coef = X \ data( :, 2 );

    %----------------------------------------------------------------------
    % 3.) coefficients of fifth-degree polynomial
    %----------------------------------------------------------------------
    % Table I fit
%     coef_1 = [ 0.140238742057073659452726133167743682861328125e4, ...
%                0.50370220225297419602838999708183109760284423828125e1, ...
%               -0.5802951355073183992150376298013725318014621734619140625e-1, ...
%                0.3318437794415666423734900813968806687626056373119354248046875e-3, ...
%               -0.144464170750563324313424373723790949952672235667705535888671875e-5, ...
%                0.2992153850746764841922537684119030865215194125994457863271236419677734375e-8 ];

	% Table II fit
%     coef_2 = [ 0.1402386826125353763927705585956573486328125e4, ...
%                0.5036721944277541496148842270486056804656982421875e1, ...
%               -0.58063786380272729148455113090676604770123958587646484375e-1, ...
%                0.3337487756955552910091700180572615863638930022716522216796875e-3, ...
%               -0.1472545242557363017862274816305312441500063869170844554901123046875e-5, ...
%                0.3114468126197573191169056490488868671473454696752014569938182830810546875e-8 ];

	% combined fit
    coef = [ 0.140238744511433878869866020977497100830078125e4, ...
             0.503715890656311326978311626589857041835784912109375e1, ...
            -0.58090835079566895127189951608670526184141635894775390625e-1, ...
             0.334402350380210116979895484945473072002641856670379638671875e-3, ...
            -0.14807647561498290376025450953978435109092970378696918487548828125e-5, ...
             0.3158928530215746731866020334078375142450312296205083839595317840576171875e-8 ];

    % (cf. Table I fit in Table III) -> Table II fit?
%     coef = [ 0.140238689e4, ...
%              0.503686088e1, ...
%             -0.580858499e-1, ...
%              0.334817140e-3, ...
%             -0.149252527e-5, ...
%              0.323913472e-8 ];

    % (cf. Table II fit in Table III) -> Table I fit?
%     coef = [ 0.140238749e4, ...
%              0.503699148e1, ...
%             -0.580268889e-1, ...
%              0.331767408e-3, ...
%             -0.144373838e-5, ...
%              0.298841057e-8 ];

    % (cf. combined fit in Table III)
%     coef = [ 0.140238754e4, ...
%              0.503711129e1, ...
%             -0.580852166e-1, ...
%              0.334198834e-8, ...
%             -0.147800417e-5, ...
%              0.314643091e-8 ];

    %----------------------------------------------------------------------
    % 4.) evaluate polynomial
    %----------------------------------------------------------------------
    exponents = ( 0:( numel( coef ) - 1 ) );
    c_water = coef( 1 ) * ones( size( T ) );
    for index_summand = 2:numel( exponents )
        c_water = c_water + coef( index_summand ) * double( T ).^exponents( index_summand );
    end

    % assign physical unit
    c_water = physical_values.meter_per_second( c_water );

end % function c_water = speed_of_sound_water( T )