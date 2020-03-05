% testbench for three-dimensional space
% material parameter: compressibility
%
% author: Martin F. Schiffner
% date: 2019-01-10
% modified: 2020-03-04

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;
clc;

addpath( genpath( './' ) );
addpath( genpath( sprintf( '/opt/matlab/R2013b/toolbox/spgl1-1.8/' ) ) );
addpath( genpath( sprintf( '/opt/matlab/R2013b/toolbox/Wavelab850/' ) ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% specify pulse-echo measurement setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% name of the simulation
%--------------------------------------------------------------------------
str_name = 'testbench_3d';

%--------------------------------------------------------------------------
% transducer array
%--------------------------------------------------------------------------
xdc_array = scattering.sequences.setups.transducers.L14_5_38;

%--------------------------------------------------------------------------
% homogeneous fluids
%--------------------------------------------------------------------------
% unperturbed mass density
rho_0 = physical_values.kilogram_per_cubicmeter( 1000 );

% time-causal absorption model
c_ref = physical_values.meter_per_second( (1480) ); %physical_values.meter_per_second( (1460:10:1520) );
f_ref = physical_values.hertz( 4e6 );
absorption_model = scattering.sequences.setups.materials.absorption_models.time_causal( zeros( size( c_ref ) ), 2.17e-3 * ones( size( c_ref ) ), 2 * ones( size( c_ref ) ), c_ref, f_ref * ones( size( c_ref ) ) );

% average group velocity
c_avg = c_ref;

% create homogeneous fluid
homogeneous_fluid = scattering.sequences.setups.materials.homogeneous_fluid( repmat( rho_0, size( c_ref ) ), absorption_model, c_avg );

%--------------------------------------------------------------------------
% field of view
%--------------------------------------------------------------------------
FOV_size_lateral = xdc_array.N_elements_axis .* xdc_array.cell_ref.edge_lengths(:);
FOV_size_axial = FOV_size_lateral( 1 );

FOV_offset_axial = physical_values.meter( 150 * 76.2e-6 );

FOV_intervals_lateral = num2cell( math.interval( - FOV_size_lateral / 2, FOV_size_lateral / 2 ) );
FOV_interval_axial = math.interval( FOV_offset_axial, FOV_offset_axial + FOV_size_axial );

FOV_cs = scattering.sequences.setups.fields_of_view.orthotope( FOV_intervals_lateral{ : }, FOV_interval_axial );

% FOV_cs = scattering.sequences.setups.fields_of_view.orthotope( math.interval( physical_values.meter(-15e-3), physical_values.meter(-5e-3) ),...
%                                                                math.interval( physical_values.meter(5e-3), physical_values.meter(10e-3) ),...
%                                                                math.interval( physical_values.meter(3e-3), physical_values.meter(43e-3) ));

%--------------------------------------------------------------------------
% create pulse-echo measurement setup
%--------------------------------------------------------------------------
setup = scattering.sequences.setups.setup( repmat( xdc_array, size( c_ref ) ), homogeneous_fluid, repmat( FOV_cs, size( c_ref ) ), repmat( { str_name }, size( c_ref ) ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% specify sequential pulse-echo measurements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) excitation parameters
%--------------------------------------------------------------------------
% specify bandwidth for the simulation
f_tx = f_ref;
frac_bw = 1;                    % fractional bandwidth of incident pulse
frac_bw_ref = -60;              % dB value that determines frac_bw
f_lb = f_tx .* ( 1 - 0.5 * frac_bw );       % lower cut-off frequency
f_ub = f_tx .* ( 1 + 0.5 * frac_bw );       % upper cut-off frequency
interval_f = math.interval( f_lb, f_ub );

% specify common excitation voltages
T_s = physical_values.second( 1 / 20e6 );
tc = gauspuls( 'cutoff', double( f_tx ), frac_bw, frac_bw_ref, -60 );     % calculate cutoff time
t = (-tc:double(T_s):tc);
pulse = gauspuls( t, double( f_tx ), frac_bw, frac_bw_ref );
axis_t = math.sequence_increasing_regular_quantized( 0, numel( t ) - 1, T_s );
u_tx_tilde = processing.signal( axis_t, physical_values.volt( pulse.' ) );

%--------------------------------------------------------------------------
% a) virtual sources of QSWs (positions and angles)
%--------------------------------------------------------------------------
N_in_per_op = 8;
N_ops = xdc_array.N_elements / N_in_per_op;

positions_src = mat2cell( [ xdc_array.positions_ctr, zeros( xdc_array.N_elements, 1 ) ], N_in_per_op * ones( 1, N_ops ), 3 );
angles = mat2cell( pi * ones( xdc_array.N_elements, 2 ) / 2, N_in_per_op * ones( 1, N_ops ), 2 );

%--------------------------------------------------------------------------
% b) directions of QPWs
%--------------------------------------------------------------------------
theta_incident = 90 * pi / 180;
% theta_incident = (77.5:2.5:102.5) * pi / 180;
e_theta = math.unit_vector( [ cos( theta_incident( : ) ), zeros( numel( theta_incident ), 1 ), sin( theta_incident( : ) ) ] );

%--------------------------------------------------------------------------
% 2.) create pulse-echo measurement sequences
%--------------------------------------------------------------------------
sequences = cell( size( setup ) );
for index_sequence = 1:numel( setup )

	% standard waves
	% sequence_PW = scattering.sequences.sequence_QPW( setup, repmat( u_tx_tilde, [3,1] ), e_theta( [1, 6, 11] ), interval_t, interval_f );
	sequences{ index_sequence } = scattering.sequences.sequence_QSW( setup( index_sequence ), repmat( u_tx_tilde, [ size( positions_src, 1 ) , 1 ] ), positions_src, angles, interval_f );
% 	sequences{ index_sequence } = scattering.sequences.sequence_QPW( setup( index_sequence ), repmat( u_tx_tilde, size( e_theta ) ), e_theta, interval_f );

	% sequence = scattering.sequences.sequence_QPW( setup, u_tx_tilde, e_theta( 1 ), interval_f );
	%         sequence = scattering.sequences.sequence_SA( setup, excitation_voltages_common, pi / 2 * ones( 128, 1 ) );
    %         settings_rng_apo = auxiliary.setting_rng( 10 * ones(11, 1), repmat({'twister'}, [ 11, 1 ]) );
    %         settings_rng_del = auxiliary.setting_rng( 3 * ones(1, 1), repmat({'twister'}, [ 1, 1 ]) );
    %         sequence = scattering.sequences.sequence_rnd_apo( setup, excitation_voltages_common, settings_rng_apo );
    %         e_dir = math.unit_vector( [ cos( 89.9 * pi / 180 ), sin( 89.9 * pi / 180 ) ] );
    %         sequence = scattering.sequences.sequence_rnd_del( setup, excitation_voltages_common, e_dir, settings_rng_del );
    %         sequence = scattering.sequences.sequence_rnd_apo_del( setup, excitation_voltages_common, e_dir, settings_rng_apo, settings_rng_del );

end

% SA sequence
sequences = cell( size( setup ) );
for index_sequence = 1:N_ops

	sequences{ index_sequence } = scattering.sequences.sequence_QSW( setup, repmat( u_tx_tilde, [ size( positions_src{ index_sequence }, 1 ) , 1 ] ),  positions_src{ index_sequence }, angles{ index_sequence }, interval_f );

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initialize scattering operators (Born approximation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TODO: turn off anti-aliasing in p_in via options

%--------------------------------------------------------------------------
% 1.) specify scattering operator options
%--------------------------------------------------------------------------
% spatial discretization options
method_faces = scattering.sequences.setups.discretizations.methods.grid_numbers( [ 4; 53 ] );
method_FOV = scattering.sequences.setups.discretizations.methods.grid_distances( physical_values.meter( [ 76.2e-6; 4e-3; 76.2e-6 ] ) );
options_disc_spatial = scattering.sequences.setups.discretizations.options( method_faces, method_FOV );

% spectral discretization options
options_disc_spectral = scattering.sequences.settings.discretizations.sequence_custom( sequences{ 1 }.interval_hull_t );

% create discretization options
options_disc = scattering.options.discretization( options_disc_spatial, options_disc_spectral );

% create static scattering operator options
options_static = scattering.options.static( options_disc );

% create momentary scattering operator options
options_momentary = scattering.options.momentary( scattering.anti_aliasing_filters.off );

% scattering options
options = scattering.options( options_static, options_momentary );

%--------------------------------------------------------------------------
% 2.) create scattering operators and linear transforms
%--------------------------------------------------------------------------
% specify cell arrays
operators_born = cell( size( sequences ) );
LTs = cell( size( sequences ) );

% iterate pulse-echo measurement sequences
for index_sequence = 1:numel( sequences )

    %----------------------------------------------------------------------
	% a) create scattering operator (Born approximation)
	%----------------------------------------------------------------------
	operators_born{ index_sequence } = scattering.operator_born( sequences{ index_sequence }, options );

    %----------------------------------------------------------------------
	% b) specify linear transforms
	%----------------------------------------------------------------------
	LTs{ index_sequence }{ 1 } = [];

end % for index_sequence = 1:numel( sequences )

% concatenate cell array contents into vector
operators_born = cat( 2, operators_born{ : } );

%--------------------------------------------------------------------------
% 3.) compute received energies
%--------------------------------------------------------------------------
% a) default settings
E_M = energy_rx( operators_born, LTs );

% ensure cell array structure
if isscalar( operators_born )
	% single operators_born && single/multiple LTs
	E_M = { E_M };
end
for index_sequence = 1:numel( sequences )
	if isscalar( LTs{ index_sequence } )
        % single/multiple operators_born && single LTs
        E_M{ index_sequence } = E_M( index_sequence );
    end
end % for index_sequence = 1:numel( sequences )

%--------------------------------------------------------------------------
% 4.) compose inverse weighting matrices and linear transforms
%--------------------------------------------------------------------------
% specify cell arrays
LT_weighting_inv = cell( size( sequences ) );

% iterate pulse-echo measurement sequences
for index_sequence = 1:numel( sequences )

	% specify cell arrays
	LT_weighting_inv{ index_sequence } = cell( size( LTs{ index_sequence } ) );

    %----------------------------------------------------------------------
	% a) canonical basis
    %----------------------------------------------------------------------
	LT_weighting_inv{ index_sequence }{ 1 } = linear_transforms.weighting( 1 ./ sqrt( double( E_M{ index_sequence }{ 1 } ) ) )';

	%----------------------------------------------------------------------
	% b) arbitrary linear transform
	%----------------------------------------------------------------------
	% iterate linear transforms
	for index_transform = 2:numel( LTs{ index_sequence } )

        % i.) default settings
        LT_weighting_inv{ index_sequence }{ index_transform } = linear_transforms.composition( linear_transforms.weighting( 1 ./ sqrt( double( E_M{ index_sequence }{ index_transform } ) ) )', LTs{ index_sequence }{ index_transform } );

	end % for index_transform = 2:numel( LTs{ index_sequence } )

end % for index_sequence = 1:numel( sequences )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% forward simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) specify coefficient vector
%--------------------------------------------------------------------------
N_coefficients = 9;
N_dimensions = operators_born( 1 ).sequence.setup.FOV.shape.grid.N_dimensions;
direction = operators_born( 1 ).sequence.setup.FOV.shape.grid.N_points_axis - ones( 1, N_dimensions );
indices_tpsf_axis = round( ones( N_coefficients, N_dimensions ) + linspace( 0.05, 0.95, N_coefficients )' * direction );
indices_tpsf = forward_index_transform( operators_born( 1 ).sequence.setup.FOV.shape.grid, indices_tpsf_axis );

theta = zeros( 512^2, 1 );
theta( indices_tpsf ) = 1;

%--------------------------------------------------------------------------
% 2.) perform forward scattering
%--------------------------------------------------------------------------
u_M = cell( 1, N_ops );

for index_sequence = 1:N_ops

	% create scattering operator
	operator = scattering.operator_born( sequences{ index_sequence }, options );

    %
    u_M{ index_sequence } = forward( operator, theta, [], scattering.options.gpu_off, scattering.anti_aliasing_filters.off );

end

u_M = cat( 1, u_M{ : } );

u_M = forward( operators_born, theta, [], scattering.options.gpu_off, scattering.anti_aliasing_filters.off );

lbs_q = zeros( size( u_M ) );
for index_sequence = 1:numel( sequences )

	temp = quantize( operators_born( index_sequence ).sequence.interval_hull_t, T_s );
    lbs_q( index_sequence ) = temp.q_lb;
end

u_M_tilde = signal( u_M, 309, T_s );

%--------------------------------------------------------------------------
% display results
%--------------------------------------------------------------------------
figure( 1 );
imagesc( illustration.dB( abs( hilbert( double( u_M_tilde( 90 ).samples ) ) ), 20 ), [ -60, 0 ] );

R = zeros( 3, 3, 2 );
factor_interp = 10;

indices_tx = [ 1, 64, 128 ];
indices_rx = indices_tx;

for index_tx = 1:numel( indices_tx )

	index_element_tx = indices_tx( index_tx );

    for index_rx = 1:numel( indices_rx )

        index_element_rx = indices_rx( index_rx );

        if index_element_rx > index_element_tx
            continue;
        end

        signal_1 = u_M_tilde( index_tx ).samples( :, index_element_rx );
        signal_2 = u_M_tilde( index_rx ).samples( :, index_element_tx );

        signal_1_int = interpft( signal_1, factor_interp * numel( signal_1 ) );
        signal_2_int = interpft( signal_2, factor_interp * numel( signal_2 ) );

        % compute inter-element correlation coefficients
        [ corr_vals, corr_lags ] = xcorr( signal_1_int / norm( signal_1_int ), signal_2_int / norm( signal_2_int ) );

        % detect and save maximum of cross-correlation
        [ ~, index_max ] = max( corr_vals );

        %
        R( index_tx, index_rx, 1 ) = corr_vals( index_max );
        R( index_tx, index_rx, 2 ) = corr_lags( index_max );

        %
        figure( 3 );
        plot( (1:factor_interp * numel( signal_1 )), signal_1_int, (1:factor_interp * numel( signal_2 )), signal_2_int );
        pause;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% adjoint simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% perform adjoint scattering
theta_hat = adjoint( operators_born, u_M, [], scattering.options.gpu_off );
time_start = tic;
for index_test = 1:1
    theta_hat_test = scattering.adjoint_quick_gpu( operators_born, repmat( double( return_vector( u_M ) ), [1,2] ), 0 );
end
time_elapsed = toc( time_start ) / 1;

theta_hat_aa = adjoint_quick( operators_born, double( return_vector( u_M ) ) );
theta_hat_weighting = adjoint_quick( operator_born, u_rx, LT_weighting );

% transform point spread functions
[ theta_hat_tpsf, E_M, adjointness ] = tpsf( operators_born, indices_tpsf, LT_weighting_inv, scattering.options.gpu_off );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calibration methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) initial states for estimate
%--------------------------------------------------------------------------
indices_t = [ 771, 899, 1028 ];
indices_elements = [  90,  107,  121 ];
c_avg_start = physical_values.meter_per_second( 1490 );

pos_start = [ xdc_array.positions_ctr( indices_elements, : ), u_M_tilde( 1 ).axis.members( indices_t( : ) ) * c_avg_start / 2 ];
% pos_start = operator.sequence.setup.FOV.shape.grid.positions( indices_tpsf(7:9), : );
states_0 = calibration.state( pos_start, c_avg_start );

% absorption model and options
handle_absorption_model = @( x ) scattering.sequences.setups.materials.absorption_models.time_causal( 0, 2.17e-3, 2, x, f_ref );
options = calibration.options( physical_values.second( 2.5e-6 ), ( numel( pulse ) - 1 ) / 2 * T_s, 1, 128, interval_f, handle_absorption_model );

[ states_est, rel_RMSE, pulse_shape_mean, pulse_shape_std_dev ] = calibration.estimate_SOS_point_qsw( u_M_tilde, xdc_array, states_0, options );

position_est = { states_est.position_target };
position_est = cat( 1, position_est{ : } );

c_avg_mean = mean( [ states_est.c_avg ] );
c_avg_std_dev = std( [ states_est.c_avg ] );

%
rel_RMSE_position = vecnorm( position_est - operator.sequence.setup.FOV.shape.grid.positions( indices_tpsf(7:9), : ), 2, 2 ) ./ vecnorm( operator.sequence.setup.FOV.shape.grid.positions( indices_tpsf(7:9), : ), 2, 2 )

%
states_updated = calibration.state( position_est, c_avg_mean );

% estimate pulse-echo responses
[ e_B_tilde, e_B_tilde_mean, e_B_tilde_std_dev ] = calibration.estimate_PER_point_qsw( u_M_tilde, xdc_array, states_updated, options );

indices_t = {...
    [ 59, 180, 301, 420, 542, 661, 781, 902, 1021 ];...
	[ 59, 178, 298, 417, 537, 656, 774, 894, 1013 ];...
    [ 59, 174, 291, 409, 527, 644, 760, 872, 987 ];...
    [ 59, 174, 291, 409, 527, 644, 760, 872, 987 ];...
    [ 59, 174, 291, 407, 524, 640, 755, 878, 994 ];...
    [ 59, 174, 291, 407, 524, 640, 755, 872, 987 ];...
	[ 59, 173, 290, 405, 521, 636, 751, 867, 982 ] };

indices_elements = {...
	[  7,  22,  36,  50,  64,  80,  94, 107,  122 ];...
    [  7,  21,  36,  50,  64,  79,  93, 108,  123 ];...
    [  7,  21,  36,  50,  65,  79,  93, 108,  122 ];...
    [  7,  21,  36,  50,  63,  79,  94, 107,  122 ];...
	[  7,  21,  36,  50,  64,  80,  94, 107,  122 ];...
	[  7,  21,  36,  50,  64,  79,  94, 107,  122 ];...
    [  7,  21,  36,  50,  64,  79,  94, 106,  122 ] };

c_avg_start = physical_values.meter_per_second( 1470 * ones( size( sequences ) ) );

% specify cell array for states_0
states_0 = cell( size( sequences ) );

% iterate pulse-echo measurement sequences
for index_sequence = 1:numel( sequences )

	% create initial states
	pos_start = [ xdc_array.positions_ctr( indices_elements{ index_sequence }, : ), u_M_tilde( index_sequence ).axis.members( indices_t{ index_sequence }( : ) ) * c_avg_start( index_sequence ) / 2 ];
%     pos_start = operators_born( index_sequence ).sequence.setup.FOV.shape.grid.positions( indices_tpsf, : );
	states_0{ index_sequence } = calibration.state( pos_start, c_avg_start( index_sequence ) );

end

%--------------------------------------------------------------------------
% 2.) options
%--------------------------------------------------------------------------
% absorption model and options
handle_absorption_model = @( x ) scattering.sequences.setups.materials.absorption_models.time_causal( 0, 2.17e-3, 2, x, f_ref );
options = calibration.options( physical_values.second( 2.5e-6 ), ( numel( pulse ) - 1 ) / 2 * T_s, 1, 128, interval_f, handle_absorption_model );

%--------------------------------------------------------------------------
% 3.) perform estimates
%--------------------------------------------------------------------------
% a) estimate speed of sound
[ states_est, rel_RMSE, pulse_shape_mean, pulse_shape_std_dev ] = calibration.estimate_SOS_point( u_M_tilde, xdc_array, states_0, options );

% b) estimate pulse-echo responses
[ e_B_tilde, e_B_tilde_mean, e_B_tilde_std_dev ] = calibration.estimate_PER_point( u_M_tilde, xdc_array, states_est, options );

%--------------------------------------------------------------------------
% 4.) compute errors
%--------------------------------------------------------------------------
% specify cell arrays
rel_RMSE_positions = cell( size( states_est ) );
rel_RMSE_c_avg = cell( size( states_est ) );
rel_RMSE_c_avg_mean = zeros( size( states_est ) );

rel_RMSE_e_B_tilde = cell( size( states_est ) );
rel_RMSE_e_B_tilde_mean = cell( size( states_est ) );

% iterate pulse-echo measurement sequences
for index_sequence = 1:numel( sequences )

	% estimated positions
	positions_est = { states_est{ index_sequence }.position_target };
	positions_est = cat( 1, positions_est{ : } );
	error_pos = positions_est - operators_born( index_sequence ).sequence.setup.FOV.shape.grid.positions( indices_tpsf, : );
	rel_RMSE_positions{ index_sequence } = vecnorm( error_pos, 2, 2 ) ./ vecnorm( operators_born( index_sequence ).sequence.setup.FOV.shape.grid.positions( indices_tpsf, : ), 2, 2 );

	% estimated SoS
	error_c_avg = [ states_est{ index_sequence }.c_avg ]' - c_ref( index_sequence );
	rel_RMSE_c_avg{ index_sequence } = norm( error_c_avg ) / c_ref( index_sequence );

	% weighted mean SoS
	weights = 1 ./ rel_RMSE{ index_sequence };
	weights = weights / sum( weights );
	c_avg_mean = [ states_est{ index_sequence }.c_avg ] * weights;
	rel_RMSE_c_avg_mean( index_sequence ) = abs( c_avg_mean - c_ref( index_sequence ) ) / c_ref( index_sequence );

	% pulse-echo responses
	rel_RMSE_e_B_tilde{ index_sequence } = cell( size( states_est{ index_sequence } ) );
	rel_RMSE_e_B_tilde_mean{ index_sequence } = zeros( size( states_est{ index_sequence } ) );

	% iterate targets
	for index_target = 1:numel( states_est{ index_sequence } )

        % ensure identical sampling periods
        if e_B_tilde{ index_sequence }{ index_target }.axis.delta ~= u_tx_tilde.axis.delta
            errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
            errorStruct.identifier = 'times_of_flight:NoSetups';
            error( errorStruct );
        end

        % common time axis
        q_lb = double( max( e_B_tilde{ index_sequence }{ index_target }.axis.q_lb, u_tx_tilde.axis.q_lb ) );
        q_ub = double( min( e_B_tilde{ index_sequence }{ index_target }.axis.q_ub, u_tx_tilde.axis.q_ub ) );
        indices_samples = (q_lb + 1):(q_ub + 1);
        axis_common = e_B_tilde{ index_sequence }{ index_target }.axis.members( indices_samples );

        % relative RMSEs of estimated pulse-echo responses
        e_B_tilde_normed = e_B_tilde{ index_sequence }{ index_target }.samples( indices_samples, : ) ./ max( e_B_tilde{ index_sequence }{ index_target }.samples( indices_samples, : ), [], 1 );
        u_tx_tilde_normed = u_tx_tilde.samples ./ max( u_tx_tilde.samples );
        error_e_B_tilde = e_B_tilde_normed - u_tx_tilde_normed;
        rel_RMSE_e_B_tilde{ index_sequence }{ index_target } = vecnorm( error_e_B_tilde, 2, 1 ) / norm( u_tx_tilde_normed );

        % relative RMSEs of mean estimated pulse-echo responses
        e_B_tilde_mean_normed = e_B_tilde_mean{ index_sequence }{ index_target }.samples( indices_samples, : ) ./ max( e_B_tilde_mean{ index_sequence }{ index_target }.samples( indices_samples, : ), [], 1 );
        error_e_B_tilde_mean = e_B_tilde_mean_normed - u_tx_tilde_normed;
        rel_RMSE_e_B_tilde_mean{ index_sequence }( index_target ) = norm( error_e_B_tilde_mean ) / norm( u_tx_tilde_normed );

        figure( index_target );
        subplot( 2, 1, 1);
        plot( e_B_tilde{ index_sequence }{ index_target }.axis.members( indices_samples ), e_B_tilde_normed, ...
              u_tx_tilde.axis.members, u_tx_tilde_normed );
        subplot( 2, 1, 2 );
        plot( e_B_tilde_mean{ index_sequence }{ index_target }.axis.members( indices_samples ), e_B_tilde_mean_normed, ...
              u_tx_tilde.axis.members, u_tx_tilde_normed, ...
              e_B_tilde_mean{ index_sequence }{ index_target }.axis.members( indices_samples ), error_e_B_tilde_mean, '--' );
        title( sprintf( 'Mean (rel. RMSE: %.2f %%)', rel_RMSE_e_B_tilde_mean{ index_sequence }( index_target ) * 1e2 ) );

    end % for index_target = 1:numel( states_est{ index_sequence } )

end % for index_sequence = 1:numel( sequences )
