% testbench for the three-dimensional space
%
% material parameter: compressibility
%
% author: Martin F. Schiffner
% date: 2019-11-16
% modified: 2020-03-04

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;
clc;

addpath( genpath( './' ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% specify pulse-echo measurement setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% name of the simulation
%--------------------------------------------------------------------------
str_name = 'testbench_3d_calibration';

%--------------------------------------------------------------------------
% 1.) transducer array
%--------------------------------------------------------------------------
xdc_array = scattering.sequences.setups.transducers.L14_5_38;

%--------------------------------------------------------------------------
% 2.) homogeneous fluids
%--------------------------------------------------------------------------
% unperturbed mass density
rho_0 = physical_values.kilogram_per_cubicmeter( 1000 );

% time-causal absorption model
c_ref = physical_values.meter_per_second( [ 1480, 1520 ] ); % physical_values.meter_per_second( (1460:10:1520) );
f_ref = physical_values.hertz( 4e6 );
absorption_model = scattering.sequences.setups.materials.absorption_models.time_causal( zeros( size( c_ref ) ), 2.17e-3 * ones( size( c_ref ) ), 2 * ones( size( c_ref ) ), c_ref, f_ref * ones( size( c_ref ) ) );

% average group velocity
c_avg = c_ref;

% create homogeneous fluid
homogeneous_fluid = scattering.sequences.setups.materials.homogeneous_fluid( repmat( rho_0, size( c_ref ) ), absorption_model, c_avg );

%--------------------------------------------------------------------------
% 3.) field of view
%--------------------------------------------------------------------------
FOV_size_lateral = xdc_array.N_elements_axis .* xdc_array.cell_ref.edge_lengths( : );
FOV_size_axial = FOV_size_lateral( 1 );

FOV_offset_axial = physical_values.meter( 360 * 76.2e-6 );

FOV_intervals_lateral = num2cell( math.interval( - FOV_size_lateral / 2, FOV_size_lateral / 2 ) );
FOV_interval_axial = math.interval( FOV_offset_axial, FOV_offset_axial + FOV_size_axial );

FOV_cs = scattering.sequences.setups.fields_of_view.orthotope( FOV_intervals_lateral{ : }, FOV_interval_axial );

%--------------------------------------------------------------------------
% 4.) create pulse-echo measurement setups
%--------------------------------------------------------------------------
setups = scattering.sequences.setups.setup( repmat( xdc_array, size( c_ref ) ), homogeneous_fluid, repmat( FOV_cs, size( c_ref ) ), repmat( { str_name }, size( c_ref ) ) );

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
% b) direction of QPW
%--------------------------------------------------------------------------
theta_incident = 90 * pi / 180;
e_theta = math.unit_vector( [ cos( theta_incident( : ) ), zeros( numel( theta_incident ), 1 ), sin( theta_incident( : ) ) ] );

%--------------------------------------------------------------------------
% 2.) create pulse-echo measurement sequences
%--------------------------------------------------------------------------
% specify cell array for sequences_SA
sequences_SA = cell( size( setups ) );
sequences_QPW = cell( size( setups ) );

% iterate setups
for index_setup = 1:numel( setups )

    % specify cell array for sequences_SA{ index_setup }
    sequences_SA{ index_setup } = cell( N_ops, 1 );

    % iterate sequences
    for index_sequence = 1:N_ops

        sequences_SA{ index_setup }{ index_sequence } = scattering.sequences.sequence_QSW( setups( index_setup ), repmat( u_tx_tilde, [ size( positions_src{ index_sequence }, 1 ) , 1 ] ),  positions_src{ index_sequence }, angles{ index_sequence }, interval_f );

    end

	% quasi-plane wave sequence
% 	sequences_QPW{ index_setup } = scattering.sequences.sequence_QPW( setups( index_setup ), repmat( u_tx_tilde, size( e_theta ) ), e_theta, interval_f );
    settings_QPW_tx = scattering.sequences.settings.controls.tx_QPW( setups( index_setup ), repmat( u_tx_tilde, size( e_theta ) ), e_theta );
	settings_QPW_rx = cell( size( settings_QPW_tx ) );
	for index_object = 1:numel( settings_QPW_tx )
        settings_QPW_rx{ index_object } = scattering.sequences.settings.controls.rx_identity( setups( index_setup ), settings_QPW_tx( index_object ), interval_f );
    end
	settings_QPW_custom = scattering.sequences.settings.setting( settings_QPW_tx, settings_QPW_rx );
	sequences_QPW{ index_setup } = scattering.sequences.sequence( setups( index_setup ), settings_QPW_custom );

end

% TODO: sequence array instead of cell array

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% forward simulations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TODO: turn off anti-aliasing in p_in via options

%--------------------------------------------------------------------------
% 2.) specify scattering operator options
%--------------------------------------------------------------------------
% spatial discretization options
method_faces = scattering.sequences.setups.discretizations.methods.grid_numbers( [ 4; 53 ] );
method_FOV = scattering.sequences.setups.discretizations.methods.grid_distances( physical_values.meter( [ 76.2e-6; 4e-3; 76.2e-6 ] ) );
options_disc_spatial = scattering.sequences.setups.discretizations.options( method_faces, method_FOV );

%--------------------------------------------------------------------------
% 3.) create scattering operators
%--------------------------------------------------------------------------
% specify cell arrays
u_M_qsw = cell( size( setups ) );
u_M_qpw = cell( size( setups ) );

u_M_tilde_qsw = cell( size( setups ) );
u_M_tilde_qpw = cell( size( setups ) );

% iterate setups
for index_setup = 1:numel( setups )

    %----------------------------------------------------------------------
    % a) specify scattering operator options
    %----------------------------------------------------------------------
	% spectral discretization options
    options_disc_spectral = scattering.sequences.settings.discretizations.sequence_custom( sequences_QPW{ index_setup }.interval_hull_t );

	% create discretization options
    options_disc = scattering.options.discretization( options_disc_spatial, options_disc_spectral );

    % create static scattering operator options
    options_static = scattering.options.static( options_disc );

    % create momentary scattering operator options
    options_momentary = scattering.options.momentary( scattering.anti_aliasing_filters.off, scattering.options.gpu_off );

	% scattering options
	options = scattering.options( options_static, options_momentary );

	% time interval
	interval_t = quantize( options_disc_spectral.interval_hull_t, T_s );

	%----------------------------------------------------------------------
	% b) QPW
	%----------------------------------------------------------------------
	% a) create scattering operator (Born approximation)
	operator_qpw = scattering.operator_born( sequences_QPW{ index_setup }, options );

    % b) specify coefficient vector
	N_coefficients = 9;
    N_dimensions = operator_qpw.sequence.setup.FOV.shape.grid.N_dimensions;
    direction = operator_qpw.sequence.setup.FOV.shape.grid.N_points_axis - ones( 1, N_dimensions );
    indices_tpsf_axis = round( ones( N_coefficients, N_dimensions ) + linspace( 0.05, 0.95, N_coefficients )' * direction );
    indices_tpsf = forward_index_transform( operator_qpw.sequence.setup.FOV.shape.grid, indices_tpsf_axis );

    theta = zeros( 512^2, 1 );
    theta( indices_tpsf ) = 1;

    % c) pulse-echo measurement
    u_M_qpw{ index_setup } = forward( operator_qpw, theta );

	% d) signals in time domain
    u_M_tilde_qpw{ index_setup } = signal( u_M_qpw{ index_setup }, double( interval_t.q_lb ), T_s );

    %----------------------------------------------------------------------
    % c) SA
    %----------------------------------------------------------------------
	% specify cell array for u_M_qsw{ index_setup }
	u_M_qsw{ index_setup } = cell( N_ops, 1 );

    % iterate sequences
    for index_sequence = 1:N_ops

        % create scattering operator (Born approximation)
        operator_qsw = scattering.operator_born( sequences_SA{ index_setup }{ index_sequence }, options );

        % pulse-echo measurement
        u_M_qsw{ index_setup }{ index_sequence } = forward( operator_qsw, theta );

    end

    % concatenate results
    u_M_qsw{ index_setup } = cat( 1, u_M_qsw{ index_setup }{ : } );

	% signals in time domain
	u_M_tilde_qsw{ index_setup } = signal( u_M_qsw{ index_setup }, double( interval_t.q_lb ), T_s );

end % for index_setup = 1:numel( setups )

% concatenate QPW results
u_M_qpw = cat( 1, u_M_qpw{ : } );
u_M_tilde_qpw = cat( 1, u_M_tilde_qpw{ : } );

%--------------------------------------------------------------------------
% display results
%--------------------------------------------------------------------------
figure( 1 );
subplot( 1, 2, 1 );
imagesc( illustration.dB( abs( hilbert( double( u_M_tilde_qpw( 1 ).samples ) ) ), 20 ), [ -60, 0 ] );
subplot( 1, 2, 2 );
imagesc( illustration.dB( abs( hilbert( double( u_M_tilde_qsw{ 1 }( 93 ).samples ) ) ), 20 ), [ -60, 0 ] );

%--------------------------------------------------------------------------
% save data for later use
%--------------------------------------------------------------------------
str_filename = sprintf( '%s_data.mat', str_name );
save( str_filename, 'u_M_qsw', 'u_M_qpw', 'u_M_tilde_qsw', 'u_M_tilde_qpw', 'indices_tpsf' );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calibration (QPW data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) initial states for estimate
%--------------------------------------------------------------------------
indices_elements = {...
% 	[  7,  22,  36,  50,  64,  80,  94, 107,  122 ];...
%     [  7,  21,  36,  50,  64,  79,  93, 108,  123 ];...
    [  7,  21,  36,  50,  65,  79,  93, 108,  122 ];...
%     [  7,  21,  36,  50,  63,  79,  94, 107,  122 ];...
% 	[  7,  21,  36,  50,  64,  80,  94, 107,  122 ];...
% 	[  7,  21,  36,  50,  64,  79,  94, 107,  122 ];...
    [  7,  21,  36,  50,  64,  79,  94, 106,  122 ] };

indices_t = {...
%     [ 59, 180, 301, 420, 542, 661, 781, 902, 1021 ];...
% 	[ 59, 178, 298, 417, 537, 656, 774, 894, 1013 ];...
    [ 59, 174, 291, 409, 527, 644, 760, 872, 987 ];...
%     [ 59, 174, 291, 409, 527, 644, 760, 872, 987 ];...
%     [ 59, 174, 291, 407, 524, 640, 755, 878, 994 ];...
%     [ 59, 174, 291, 407, 524, 640, 755, 872, 987 ];...
	[ 59, 173, 290, 405, 521, 636, 751, 867, 982 ] };

c_avg_start = c_ref - physical_values.meter_per_second( 5 );

% specify cell array for states_0
states_0 = cell( size( setups ) );

% iterate pulse-echo measurement setups
for index_setup = 1:numel( setups )

	% create initial states
	pos_start = [ xdc_array.positions_ctr( indices_elements{ index_setup }, : ), u_M_tilde_qpw( index_setup ).axis.members( indices_t{ index_setup }( : ) ) * c_avg_start( index_setup ) / 2 ];
%     pos_start = operators_born( index_setup ).sequence.setup.FOV.shape.grid.positions( indices_tpsf, : );
	states_0{ index_setup } = calibration.state( pos_start, c_avg_start( index_setup ) );

end

%--------------------------------------------------------------------------
% 2.) options
%--------------------------------------------------------------------------
% create sound speed estimation options
options_SoS = calibration.options.SoS( physical_values.second( 2.5e-6 ), ( numel( pulse ) - 1 ) / 2 * T_s, (1:xdc_array.N_elements), 1, xdc_array.N_elements );

% create pulse-echo response estimation options
handle_absorption_model = @( x ) scattering.sequences.setups.materials.absorption_models.time_causal( 0, 2.17e-3, 2, x, f_ref );
options_PER = calibration.options.per_qsw( physical_values.second( 2.5e-6 ), (1:xdc_array.N_elements), (1:xdc_array.N_elements), interval_f, handle_absorption_model );

%--------------------------------------------------------------------------
% 3.) perform estimates
%--------------------------------------------------------------------------
% a) estimate speed of sound
[ states_est, rel_RMSE, pulse_shape_mean, pulse_shape_std_dev ] = calibration.estimate_SOS_point( u_M_tilde_qpw, xdc_array, states_0', options );

% iterate pulse-echo measurement sequences
states_updated = states_est;
for index_setup = 1:numel( setups )

	% estimated positions
	positions_est = { states_est{ index_setup }.position_target };
	positions_est = cat( 1, positions_est{ : } );

	% estimated speeds of sound
	c_avg_mean = mean( [ states_est{ index_setup }.c_avg ] );

	% updated states
	states_updated{ index_setup } = calibration.state( positions_est, c_avg_mean );

end

% b) estimate pulse-echo responses
[ e_B_tilde, e_B_tilde_mean, e_B_tilde_std_dev ] = calibration.estimate_PER_point( u_M_tilde_qpw, xdc_array, states_updated, options );

%--------------------------------------------------------------------------
% 4.) compute errors
%--------------------------------------------------------------------------
% specify cell arrays
rel_RMSE_positions = cell( size( setups ) );
rel_RMSE_c_avg = cell( size( setups ) );
rel_RMSE_c_avg_mean = zeros( size( setups ) );

rel_RMSE_e_B_tilde = cell( size( setups ) );
rel_RMSE_e_B_tilde_mean = cell( size( setups ) );
rho_e_B_tilde = cell( size( setups ) );
rho_e_B_tilde_mean = cell( size( setups ) );

% interpolate tx voltage for cross-correlation
u_tx_tilde_int = interpolate( u_tx_tilde, 30 );
u_tx_tilde_int_normed = u_tx_tilde_int.samples ./ norm( u_tx_tilde_int.samples );

% iterate pulse-echo measurement sequences
for index_setup = 1:numel( setups )

	% estimated positions
	positions_est = { states_est{ index_setup }.position_target };
	positions_est = cat( 1, positions_est{ : } );
	error_pos = positions_est - operator_qpw.sequence.setup.FOV.shape.grid.positions( indices_tpsf, : );
	rel_RMSE_positions{ index_setup } = vecnorm( error_pos, 2, 2 ) ./ vecnorm( operator_qpw.sequence.setup.FOV.shape.grid.positions( indices_tpsf, : ), 2, 2 );

	% estimated SoS
	error_c_avg = [ states_est{ index_setup }.c_avg ]' - c_ref( index_setup );
	rel_RMSE_c_avg{ index_setup } = norm( error_c_avg ) / c_ref( index_setup );

	% weighted mean SoS
	weights = 1 ./ rel_RMSE{ index_setup };
	weights = weights / sum( weights );
	c_avg_mean = [ states_est{ index_setup }.c_avg ] * weights;
	rel_RMSE_c_avg_mean( index_setup ) = abs( c_avg_mean - c_ref( index_setup ) ) / c_ref( index_setup );

	% pulse-echo responses
	rel_RMSE_e_B_tilde{ index_setup } = cell( size( states_est{ index_setup } ) );
	rel_RMSE_e_B_tilde_mean{ index_setup } = zeros( size( states_est{ index_setup } ) );
    rho_e_B_tilde{ index_setup } = zeros( size( states_est{ index_setup } ) );
    rho_e_B_tilde_mean{ index_setup } = zeros( size( states_est{ index_setup } ) );

	% iterate targets
	for index_target = 1:numel( states_est{ index_setup } )

        % ensure identical sampling periods
        if e_B_tilde{ index_setup }( index_target ).axis.delta ~= u_tx_tilde.axis.delta
            errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
            errorStruct.identifier = 'times_of_flight:NoSetups';
            error( errorStruct );
        end

        % interpolate PE responses and normalize
        e_B_tilde_int = interpolate( e_B_tilde{ index_setup }( index_target ), 30 );
        e_B_tilde_int_normed = e_B_tilde_int.samples ./ vecnorm( e_B_tilde_int.samples, 2, 1 );
        e_B_tilde_mean_int = interpolate( e_B_tilde_mean{ index_setup }( index_target ), 30 );
        e_B_tilde_mean_int_normed = e_B_tilde_mean_int.samples ./ vecnorm( e_B_tilde_mean_int.samples, 2, 1 );

        % cross-correlation
        [ corr_vals, corr_lags ] = xcorr( e_B_tilde_int_normed( :, 1 ), u_tx_tilde_int_normed );
        [ rho_e_B_tilde{ index_setup }( index_target ), index_max ] = max( corr_vals );
        [ corr_vals_mean, corr_lags_mean ] = xcorr( e_B_tilde_mean_int_normed, u_tx_tilde_int_normed );
        [ rho_e_B_tilde_mean{ index_setup }( index_target ), index_max_mean ] = max( corr_vals_mean );

        % shift PE responses
        axis_shifted = math.sequence_increasing_regular_quantized( e_B_tilde_int.axis.q_lb - corr_lags( index_max ), e_B_tilde_int.axis.q_ub - corr_lags( index_max ), e_B_tilde_int.axis.delta );
        e_B_tilde_int_shifted = processing.signal_matrix( axis_shifted, e_B_tilde_int.samples );
        axis_shifted = math.sequence_increasing_regular_quantized( e_B_tilde_mean_int.axis.q_lb - corr_lags_mean( index_max_mean ), e_B_tilde_mean_int.axis.q_ub - corr_lags_mean( index_max_mean ), e_B_tilde_mean_int.axis.delta );
        e_B_tilde_mean_int_shifted = processing.signal_matrix( axis_shifted, e_B_tilde_mean_int.samples );

        % common time axis
%         q_lb = double( max( e_B_tilde_mean_int_shifted.axis.q_lb, u_tx_tilde_int.axis.q_lb ) );
%         q_ub = double( min( e_B_tilde_mean_int_shifted.axis.q_ub, u_tx_tilde_int.axis.q_ub ) );
        
%         indices_samples = (q_lb + 1):(q_ub + 1);
%         axis_common = e_B_tilde_int.axis.members( indices_samples );

%         error_e_B_tilde = e_B_tilde_int_normed - u_tx_tilde_int_normed;
%         rel_RMSE_e_B_tilde{ index_setup }{ index_target } = vecnorm( error_e_B_tilde, 2, 1 ) / norm( u_tx_tilde_int_normed );

        % relative RMSEs of mean estimated pulse-echo responses
%         e_B_tilde_mean_normed = e_B_tilde_mean{ index_setup }( index_target ).samples( indices_samples, : ) ./ max( e_B_tilde_mean{ index_setup }( index_target ).samples( indices_samples, : ), [], 1 );
%         error_e_B_tilde_mean = e_B_tilde_mean_normed - u_tx_tilde_int_normed;
%         rel_RMSE_e_B_tilde_mean{ index_setup }( index_target ) = norm( error_e_B_tilde_mean ) / norm( u_tx_tilde_int_normed );

        % illustrate
        figure( index_target );
        subplot( 2, 1, 1 );
        plot( e_B_tilde_int_shifted.axis.members, e_B_tilde_int_normed, ...
              u_tx_tilde_int.axis.members, u_tx_tilde_int_normed );
        subplot( 2, 1, 2 );
        plot( e_B_tilde_mean_int_shifted.axis.members, e_B_tilde_mean_int_normed, ...
              u_tx_tilde_int.axis.members, u_tx_tilde_int_normed );%, ...
%               e_B_tilde_mean_sa{ index_setup }( index_target ).axis.members( indices_samples ), error_e_B_tilde_mean, '--' );
%         title( sprintf( 'Mean (rel. RMSE: %.2f %%)', rel_RMSE_e_B_tilde_mean_sa{ index_setup }( index_target ) * 1e2 ) );

    end % for index_target = 1:numel( states_est{ index_setup } )

end % for index_setup = 1:numel( setups )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calibration (QSW data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) initial states for estimate
%--------------------------------------------------------------------------
indices_elements_sa = {...
% 	[  7,  22,  36,  50,  64,  80,  94, 107,  122 ];...
%     [  7,  21,  36,  50,  64,  79,  93, 108,  123 ];...
    [  93, 108,  122 ];...
%     [  7,  21,  36,  50,  63,  79,  94, 107,  122 ];...
% 	[  7,  21,  36,  50,  64,  80,  94, 107,  122 ];...
% 	[  7,  21,  36,  50,  64,  79,  94, 107,  122 ];...
    [  93, 107,  122 ] };

indices_t_sa = {...
%     [ 59, 180, 301, 420, 542, 661, 781, 902, 1021 ];...
% 	[ 59, 178, 298, 417, 537, 656, 774, 894, 1013 ];...
    [ 767, 887, 1004 ];...
%     [ 59, 174, 291, 409, 527, 644, 760, 872, 987 ];...
%     [ 59, 174, 291, 407, 524, 640, 755, 878, 994 ];...
%     [ 59, 174, 291, 407, 524, 640, 755, 872, 987 ];...
	[ 747, 863, 977 ] };

c_avg_start_sa = c_ref - physical_values.meter_per_second( 10 );

% specify cell array for states_0
states_0_sa = cell( size( setups ) );

% iterate pulse-echo measurement sequences
for index_setup = 1:numel( setups )

	% create initial states
	pos_start_sa = [ xdc_array.positions_ctr( indices_elements_sa{ index_setup }, : ), u_M_tilde_qsw{ index_setup }( 1 ).axis.members( indices_t_sa{ index_setup }( : ) ) * c_avg_start_sa( index_setup ) / 2 ];
% 	pos_start_sa = operator_qsw.sequence.setup.FOV.shape.grid.positions( indices_tpsf( 6:9 ), : );
	states_0_sa{ index_setup } = calibration.state( pos_start_sa, c_avg_start_sa( index_setup ) );

end

%--------------------------------------------------------------------------
% 2.) options
%--------------------------------------------------------------------------
% create sound speed estimation options
options_sos_qsw = calibration.options.sos_qsw( physical_values.second( 2.5e-6 ), ( numel( pulse ) - 1 ) / 2 * T_s, (1:xdc_array.N_elements), 1, xdc_array.N_elements, 60, auxiliary.setting_window( @tukeywin, 0.1 ) );
options_sos_focus = calibration.options.sos_focus( physical_values.second( 2.5e-6 ), ( numel( pulse ) - 1 ) / 2 * T_s, interval_f, scattering.anti_aliasing_filters.raised_cosine( 0.25 ), 0.4, 60, 20, 1e-3 );

% create pulse-echo response estimation options
handle_absorption_model_sa = @( x ) scattering.sequences.setups.materials.absorption_models.time_causal( 0, 2.17e-3, 2, x, f_ref );
options_PER_sa = calibration.options.per_qsw( physical_values.second( 2.5e-6 ), (1:xdc_array.N_elements), (1:xdc_array.N_elements), interval_f, handle_absorption_model_sa );

%--------------------------------------------------------------------------
% 3.) perform estimates
%--------------------------------------------------------------------------
% a) estimate speed of sound
[ states_est_qsw, rel_RMSE_qsw ] = calibration.estimate_sos_point_qsw( u_M_tilde_qsw, xdc_array, states_0_sa, options_sos_qsw );
[ states_est_focus, rel_RMSE_focus ] = calibration.estimate_sos_point_focus( u_M_tilde_qsw, xdc_array, states_0_sa, options_sos_focus );

% ensure cell arrays

% iterate pulse-echo measurement setups
states_updated_sa = states_est_qsw;
for index_setup = 1:numel( setups )

	% estimated positions
	positions_est_sa = { states_est_qsw{ index_setup }.position_target };
	positions_est_sa = cat( 1, positions_est_sa{ : } );

	% estimated speed of sound
	c_avg_mean_sa = mean( [ states_est_qsw{ index_setup }.c_avg ] );

	% updated states
    states_updated_sa{ index_setup } = calibration.state( positions_est_sa, c_avg_mean_sa );

end

% b) estimate pulse-echo responses
[ e_B_tilde_ref_sa, cal_tx_tilde_sa, cal_rx_tilde_sa, rel_RMSE_local_sa, e_B_tilde_sa, e_B_tilde_mean_sa, e_B_tilde_std_dev_sa ] = calibration.estimate_PER_point_qsw( u_M_tilde_qsw, xdc_array, states_updated_sa, options_PER_sa );

%--------------------------------------------------------------------------
% 4.) compute errors
%--------------------------------------------------------------------------
%
index_target_lb = 7;
index_target_ub = 9;

% specify cell arrays
rel_RMSE_positions_qsw = cell( size( setups ) );
rel_RMSE_c_avg_qsw = cell( size( setups ) );
rel_RMSE_c_avg_mean_qsw = zeros( size( setups ) );

rel_RMSE_positions_focus = cell( size( setups ) );
rel_RMSE_c_avg_focus = cell( size( setups ) );
rel_RMSE_c_avg_mean_focus = zeros( size( setups ) );

rel_RMSE_e_B_tilde_qsw = cell( size( setups ) );
rel_RMSE_e_B_tilde_mean_qsw = cell( size( setups ) );
rho_e_B_tilde_qsw = cell( size( setups ) );
rho_e_B_tilde_mean_qsw = cell( size( setups ) );

% interpolate tx voltage for cross-correlation
u_tx_tilde_int = interpolate( u_tx_tilde, 30 );
u_tx_tilde_int_normed = u_tx_tilde_int.samples ./ norm( u_tx_tilde_int.samples );

% iterate pulse-echo measurement sequences
for index_setup = 1:numel( setups )

    %----------------------------------------------------------------------
	% a) estimated positions
    %----------------------------------------------------------------------
    % a) QSW model
	positions_est_qsw = { states_est_qsw{ index_setup }.position_target };
	positions_est_qsw = cat( 1, positions_est_qsw{ : } );
	error_pos = positions_est_qsw - operator_qsw.sequence.setup.FOV.shape.grid.positions( indices_tpsf( index_target_lb:index_target_ub ), : );
	rel_RMSE_positions_qsw{ index_setup } = vecnorm( error_pos, 2, 2 ) ./ vecnorm( operator_qsw.sequence.setup.FOV.shape.grid.positions( indices_tpsf( index_target_lb:index_target_ub ), : ), 2, 2 );

	% b) refocusing
	positions_est_focus = { states_est_focus{ index_setup }.position_target };
	positions_est_focus = cat( 1, positions_est_focus{ : } );
	error_pos = positions_est_focus - operator_qsw.sequence.setup.FOV.shape.grid.positions( indices_tpsf( index_target_lb:index_target_ub ), : );
	rel_RMSE_positions_focus{ index_setup } = vecnorm( error_pos, 2, 2 ) ./ vecnorm( operator_qsw.sequence.setup.FOV.shape.grid.positions( indices_tpsf( index_target_lb:index_target_ub ), : ), 2, 2 );

    %----------------------------------------------------------------------
	% b) estimated SoS
    %----------------------------------------------------------------------
	% i.) QSW model
	error_c_avg = [ states_est_qsw{ index_setup }.c_avg ]' - c_ref( index_setup );
	rel_RMSE_c_avg_qsw{ index_setup } = norm( error_c_avg ) / c_ref( index_setup );

    % ii.) QSW model (weighted)
	weights = 1 ./ rel_RMSE_qsw{ index_setup };
	weights = weights / sum( weights );
	c_avg_mean_qsw = [ states_est_qsw{ index_setup }.c_avg ] * weights;
	rel_RMSE_c_avg_mean_qsw( index_setup ) = abs( c_avg_mean_qsw - c_ref( index_setup ) ) / c_ref( index_setup );

	% iii.) refocusing
	error_c_avg = [ states_est_focus{ index_setup }.c_avg ]' - c_ref( index_setup );
	rel_RMSE_c_avg_focus{ index_setup } = norm( error_c_avg ) / c_ref( index_setup );

	% iv.) refocusing (weighted)
	weights = 1 ./ rel_RMSE_focus{ index_setup };
	weights = weights / sum( weights );
	c_avg_mean_focus = [ states_est_focus{ index_setup }.c_avg ] * weights;
	rel_RMSE_c_avg_mean_focus( index_setup ) = abs( c_avg_mean_focus - c_ref( index_setup ) ) / c_ref( index_setup );

    %----------------------------------------------------------------------
	% c) pulse-echo responses
    %----------------------------------------------------------------------
% 	rel_RMSE_e_B_tilde_sa{ index_setup } = cell( size( states_est_qsw{ index_setup } ) );
% 	rel_RMSE_e_B_tilde_mean_sa{ index_setup } = zeros( size( states_est_qsw{ index_setup } ) );
% 
% 	% iterate targets
% 	for index_target = 1:numel( states_est_qsw{ index_setup } )
% 
%         % ensure identical sampling periods
%         if e_B_tilde_ref_sa{ index_setup }( index_target ).axis.delta ~= u_tx_tilde.axis.delta
%             errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
%             errorStruct.identifier = 'times_of_flight:NoSetups';
%             error( errorStruct );
%         end
% 
%         % interpolate PE responses and normalize
%         e_B_tilde_int = interpolate( e_B_tilde_ref_sa{ index_setup }( index_target ), 30 );
%         e_B_tilde_int_normed = e_B_tilde_int.samples ./ vecnorm( e_B_tilde_int.samples, 2, 1 );
%         e_B_tilde_mean_int = interpolate( e_B_tilde_mean_sa{ index_setup }( index_target ), 30 );
%         e_B_tilde_mean_int_normed = e_B_tilde_mean_int.samples ./ vecnorm( e_B_tilde_mean_int.samples, 2, 1 );
% 
%         % cross-correlation
%         [ corr_vals, corr_lags ] = xcorr( e_B_tilde_int_normed, u_tx_tilde_int_normed );
%         [ rho_e_B_tilde_sa{ index_setup }( index_target ), index_max ] = max( corr_vals );
%         [ corr_vals_mean, corr_lags_mean ] = xcorr( e_B_tilde_mean_int_normed, u_tx_tilde_int_normed );
%         [ rho_e_B_tilde_mean_sa{ index_setup }( index_target ), index_max_mean ] = max( corr_vals_mean );
% 
%         % shift PE responses
%         axis_shifted = math.sequence_increasing_regular_quantized( e_B_tilde_int.axis.q_lb - corr_lags( index_max ), e_B_tilde_int.axis.q_ub - corr_lags( index_max ), e_B_tilde_int.axis.delta );
%         e_B_tilde_int_shifted = processing.signal_matrix( axis_shifted, e_B_tilde_int.samples );
%         axis_shifted = math.sequence_increasing_regular_quantized( e_B_tilde_mean_int.axis.q_lb - corr_lags_mean( index_max_mean ), e_B_tilde_mean_int.axis.q_ub - corr_lags_mean( index_max_mean ), e_B_tilde_mean_int.axis.delta );
%         e_B_tilde_mean_int_shifted = processing.signal_matrix( axis_shifted, e_B_tilde_mean_int.samples );
% 
%         % common time axis
% %         q_lb = double( max( e_B_tilde_int.axis.q_lb - corr_lags( index_max ), u_tx_tilde_int.axis.q_lb ) );
% %         q_ub = double( min( e_B_tilde_int.axis.q_ub - corr_lags( index_max ), u_tx_tilde_int.axis.q_ub ) );
% %         indices_samples = (q_lb + 1):(q_ub + 1);
% %         axis_common = e_B_tilde_int.axis.members( indices_samples );
% 
%         % relative RMSEs of estimated pulse-echo responses
% %         error_e_B_tilde = e_B_tilde_int_normed( indices_samples ) - u_tx_tilde_int_normed;
% %         rel_RMSE_e_B_tilde_sa{ index_setup }{ index_target } = vecnorm( error_e_B_tilde, 2, 1 ) / norm( u_tx_tilde_int_normed );
% 
%         % relative RMSEs of mean estimated pulse-echo responses
% %         e_B_tilde_mean_normed = e_B_tilde_mean_sa{ index_setup }( index_target ).samples( indices_samples, : ) ./ max( e_B_tilde_mean_sa{ index_setup }( index_target ).samples( indices_samples, : ), [], 1 );
% %         error_e_B_tilde_mean = e_B_tilde_mean_normed - u_tx_tilde_int_normed;
% %         rel_RMSE_e_B_tilde_mean_sa{ index_setup }( index_target ) = norm( error_e_B_tilde_mean ) / norm( u_tx_tilde_int_normed );
% 
%         % illustrate
%         figure( index_target );
%         subplot( 2, 1, 1 );
%         plot( e_B_tilde_int_shifted.axis.members, e_B_tilde_int_normed, ...
%               u_tx_tilde_int.axis.members, u_tx_tilde_int_normed );
%         subplot( 2, 1, 2 );
%         plot( e_B_tilde_mean_int_shifted.axis.members, e_B_tilde_mean_int_normed, ...
%               u_tx_tilde_int.axis.members, u_tx_tilde_int_normed );%, ...
% %               e_B_tilde_mean_sa{ index_setup }( index_target ).axis.members( indices_samples ), error_e_B_tilde_mean, '--' );
% %         title( sprintf( 'Mean (rel. RMSE: %.2f %%)', rel_RMSE_e_B_tilde_mean_sa{ index_setup }( index_target ) * 1e2 ) );
% 
%     end % for index_target = 1:numel( states_est_qsw{ index_setup } )

end % for index_setup = 1:numel( setups )
