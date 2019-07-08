% reconstruct object A from simulated RF signals
% material parameter: compressibility
%
% author: Martin F. Schiffner
% date: 2019-01-10
% modified: 2019-06-18

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
str_name = 'wire_phantom';

%--------------------------------------------------------------------------
% transducer array
%--------------------------------------------------------------------------
xdc_array = transducers.L14_5_38( 2 );

%--------------------------------------------------------------------------
% homogeneous fluids
%--------------------------------------------------------------------------
% unperturbed mass density
rho_0 = physical_values.kilogram_per_cubicmeter( 1000 );

% time-causal absorption model
c_ref = physical_values.meter_per_second( (1480:5:1520) );
f_ref = physical_values.hertz( 4e6 );
absorption_model = absorption_models.time_causal( zeros( size( c_ref ) ), 2.17e-3 * ones( size( c_ref ) ), 2 * ones( size( c_ref ) ), c_ref, f_ref * ones( size( c_ref ) ) );

% average group velocity
c_avg = c_ref;

% create homogeneous fluid
homogeneous_fluid = pulse_echo_measurements.homogeneous_fluid( repmat( rho_0, size( c_ref ) ), absorption_model, c_avg );

%--------------------------------------------------------------------------
% field of view
%--------------------------------------------------------------------------
FOV_size_lateral = xdc_array.N_elements_axis .* xdc_array.cell_ref.edge_lengths;
FOV_size_axial = FOV_size_lateral( 1 );

FOV_offset_axial = physical_values.meter( 350 * 76.2e-6 );

FOV_intervals_lateral = num2cell( math.interval( - FOV_size_lateral / 2, FOV_size_lateral / 2 ) );
FOV_interval_axial = math.interval( FOV_offset_axial, FOV_offset_axial + FOV_size_axial );

FOV_cs = fields_of_view.orthotope( FOV_intervals_lateral{ : }, FOV_interval_axial );

%--------------------------------------------------------------------------
% create pulse-echo measurement setup
%--------------------------------------------------------------------------
setup = pulse_echo_measurements.setup( repmat( xdc_array, size( c_ref ) ), homogeneous_fluid, repmat( FOV_cs, size( c_ref ) ), repmat( { str_name }, size( c_ref ) ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% specify sequential pulse-echo measurements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) excitation parameters
%--------------------------------------------------------------------------
% specify bandwidth for the simulation
f_tx = f_ref;
frac_bw = 0.7;                  % fractional bandwidth of incident pulse
frac_bw_ref = -60;              % dB value that determines frac_bw
f_lb = f_tx .* ( 1 - 0.5 * frac_bw );       % lower cut-off frequency
f_ub = f_tx .* ( 1 + 0.5 * frac_bw );       % upper cut-off frequency
interval_f = math.interval( f_lb, f_ub );

% specify common excitation voltages
T_s = physical_values.second( 1 / 20e6 );
tc = gauspuls( 'cutoff', double( f_tx ), frac_bw, frac_bw_ref, -60 );     % calculate cutoff time
t = (-tc:double(T_s):tc);
pulse = gauspuls( t, double( f_tx ), frac_bw, frac_bw_ref );
axis_t = math.sequence_increasing_regular( 0, numel( t ) - 1, T_s );
u_tx_tilde = discretizations.signal( axis_t, physical_values.volt( pulse.' ) );

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
    % sequence_PW = pulse_echo_measurements.sequence_QPW( setup, repmat( u_tx_tilde, [3,1] ), e_theta( [1, 6, 11] ), interval_t, interval_f );
    % sequence_QSW = pulse_echo_measurements.sequence_QSW( setup, repmat( u_tx_tilde, size( positions_src, 1 ) ), positions_src, angles, data_RF_qsw.interval_f );
    sequences{ index_sequence } = pulse_echo_measurements.sequence_QPW( setup( index_sequence ), repmat( u_tx_tilde, size( e_theta ) ), e_theta, interval_f );

    % sequence = pulse_echo_measurements.sequence_QPW( setup, u_tx_tilde, e_theta( 1 ), interval_f );
    %         sequence = pulse_echo_measurements.sequence_SA( setup, excitation_voltages_common, pi / 2 * ones( 128, 1 ) );
    %         settings_rng_apo = auxiliary.setting_rng( 10 * ones(11, 1), repmat({'twister'}, [ 11, 1 ]) );
    %         settings_rng_del = auxiliary.setting_rng( 3 * ones(1, 1), repmat({'twister'}, [ 1, 1 ]) );
    %         sequence = pulse_echo_measurements.sequence_rnd_apo( setup, excitation_voltages_common, settings_rng_apo );
    %         e_dir = math.unit_vector( [ cos( 89.9 * pi / 180 ), sin( 89.9 * pi / 180 ) ] );
    %         sequence = pulse_echo_measurements.sequence_rnd_del( setup, excitation_voltages_common, e_dir, settings_rng_del );
    %         sequence = pulse_echo_measurements.sequence_rnd_apo_del( setup, excitation_voltages_common, e_dir, settings_rng_apo, settings_rng_del );

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initialize scattering operators (Born approximation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) specify options
%--------------------------------------------------------------------------
% spatial discretization options
parameters_elements = discretizations.parameters_number( [ 4, 53 ] );
parameters_FOV = discretizations.parameters_distance( physical_values.meter( [ 76.2e-6, 4e-3, 76.2e-6 ] ) );
options_disc_spatial = discretizations.options_spatial_grid( parameters_elements, parameters_FOV );

% spectral discretization options
% options_disc_spectral = discretizations.options_spectral.signal;
% options_disc_spectral = discretizations.options_spectral.setting;
options_disc_spectral = discretizations.options_spectral.sequence;

% create discretization options
options_disc = discretizations.options( options_disc_spatial, options_disc_spectral );

% scattering options
options = scattering.options( options_disc );

% specify recording time interval
% t_lb = 0 .* T_s;        % lower cut-off time
% t_ub = 4 * 1700 .* T_s;     % upper cut-off time
% interval_t = math.interval( t_lb, t_ub );

%--------------------------------------------------------------------------
% 2.) create scattering operators
%--------------------------------------------------------------------------
% specify cell arrays
operators_born = cell( size( sequences ) );
LT_weighting = cell( size( sequences ) );

% iterate pulse-echo measurement sequences
for index_sequence = 1:numel( sequences )

	%----------------------------------------------------------------------
	% a) initialize scattering operator (Born approximation)
	%----------------------------------------------------------------------
	operators_born{ index_sequence } = scattering.operator_born( sequences{ index_sequence }, options );

    %----------------------------------------------------------------------
    % b) specify transform and weighting matrices
    %----------------------------------------------------------------------
    % threshold for weighting
%     E_M_rnd_apo = operator_born_rnd_apo.E_M;
%     E_M_rnd_apo_max = max( E_M_rnd_apo );
%     E_M_threshold = E_M_rnd_apo_max * 0;
%     indicator = E_M_rnd_apo < E_M_threshold;
%     N_cols_threshold_kappa_id = sum( indicator(:) );
%     E_M_rnd_apo( indicator ) = E_M_threshold;
	LT_weighting{ index_sequence } = linear_transforms.weighting( 1 ./ sqrt( double( operators_born{ index_sequence }.E_M ) ) );

end % for index_sequence = 1:numel( sequences )

operators_born = cat( 2, operators_born{ : } );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% forward simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) specify coefficient vector
%--------------------------------------------------------------------------
N_coefficients = 9;
N_dimensions = operators_born( 1 ).discretization.spatial.grid_FOV.N_dimensions;
direction = operators_born( 1 ).discretization.spatial.grid_FOV.N_points_axis - ones( 1, N_dimensions );
indices_tpsf_axis = round( ones( N_coefficients, N_dimensions ) + linspace( 0.05, 0.95, N_coefficients )' * direction );
indices_tpsf = forward_index_transform( operators_born( 1 ).discretization.spatial.grid_FOV, indices_tpsf_axis );

theta = zeros( 512^2, 1 );
theta( indices_tpsf ) = 1;

%--------------------------------------------------------------------------
% 2.) perform forward scattering
%--------------------------------------------------------------------------
u_M = forward( operators_born, theta );

lbs_q = zeros( size( u_M ) );
for index_sequence = 1:numel( sequences )

	temp = quantize( operators_born( index_sequence ).sequence.interval_hull_t, T_s );
    lbs_q( index_sequence ) = temp.q_lb;
end

u_M_tilde = signal( [u_M{:}], lbs_q, T_s );

% perform adjoint scattering
theta_hat = adjoint_quick( operators_born, double( return_vector( u_M ) ) );
time_start = tic;
for index_test = 1:1
    theta_hat_test = scattering.adjoint_quick_gpu( operators_born, repmat( double( return_vector( u_M ) ), [1,2] ), 0 );
end
time_elapsed = toc( time_start ) / 1;


theta_hat_aa = adjoint_quick( operators_born, double( return_vector( u_M ) ) );
theta_hat_weighting = adjoint_quick( operator_born, u_rx, LT_weighting );

% transform point spread functions
[ theta_hat_tpsf, E_rx, adjointness ] = tpsf( operators_born, indices_tpsf, LT_weighting{1} );
[ theta_hat_tpsf_aa, E_rx_aa, adjointness_aa ] = tpsf( operators_born, indices_tpsf, LT_weighting{1} );

% u_rx_tilde = signal( u_rx, 0, T_s );

%--------------------------------------------------------------------------
% display results
%--------------------------------------------------------------------------
figure( 1 );
subplot( 1, 2, 1 );
imagesc( double( u_M_tilde( 1 ).samples ) );
subplot( 1, 2, 2 );
imagesc( illustration.dB( abs( hilbert( double( u_M_tilde( 1 ).samples ) ) ), 20 ), [ -20, 0 ] );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% test calibration procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) initial states for estimate
%--------------------------------------------------------------------------
indices_t = {...
	[ 59, 176, 295, 413, 532, 648, 767, 886, 1004 ];...
	[ 59, 176, 295, 413, 532, 648, 767, 886, 1004 ];...
    [ 59, 175, 292, 410, 529, 645, 761, 880,  997 ];...
    [ 59, 175, 292, 410, 529, 645, 761, 880,  997 ];...
    [ 57, 172, 290, 406, 523, 639, 755, 872,  989 ];...
    [ 58, 173, 290, 406, 523, 639, 754, 871,  987 ];...
	[ 57, 172, 289, 404, 521, 636, 751, 868,  984 ];...
    [ 57, 171, 288, 403, 519, 634, 748, 865,  980 ];...
	[ 57, 171, 287, 401, 517, 631, 746, 861,  976 ] };

indices_elements = {...
	[  7,  22,  36,  46,  64,  79,  89, 109,  117 ];...
    [  7,  22,  36,  46,  64,  79,  89, 109,  117 ];...
    [  7,  21,  36,  46,  63,  79,  93, 107,  117 ];...
    [  7,  21,  36,  46,  63,  79,  93, 107,  117 ];...
	[  7,  21,  36,  46,  64,  80,  93, 107,  117 ];...
	[  7,  21,  36,  46,  65,  79,  91, 108,  117 ];...
    [  7,  21,  36,  47,  64,  79,  94, 107,  117 ];...
    [  7,  21,  36,  47,  64,  79,  94, 107,  117 ];...
    [  7,  21,  36,  47,  64,  79,  94, 107,  117 ] };

c_avg_start = physical_values.meter_per_second( 1488 * ones( size( sequences ) ) );

% specify cell array for states_0
states_0 = cell( size( sequences ) );

% iterate pulse-echo measurement sequences
for index_sequence = 1:numel( sequences )

	% create initial states
	pos_start = [ xdc_array.positions_ctr( indices_elements{ index_sequence }, 1:2 ), u_M_tilde( index_sequence ).axis.members( indices_t{ index_sequence }( : ) ) * c_avg_start( index_sequence ) / 2 ];
	states_0{ index_sequence } = calibration.state( pos_start, c_avg_start( index_sequence ) );

end

%--------------------------------------------------------------------------
% 2.) options
%--------------------------------------------------------------------------
% absorption model
handle_absorption_model = @( x ) absorption_models.time_causal( 0, 2.17e-3, 2, x, f_ref );
options = calibration.options( physical_values.second( 3e-6 ), 31 * T_s, 1, 128, interval_f, handle_absorption_model );

%--------------------------------------------------------------------------
% 3.) perform estimates
%--------------------------------------------------------------------------
% estimate speed of sound
[ states_est, rel_RMSE, pulse_shape_mean, pulse_shape_std_dev ] = calibration.estimate_SOS_point( u_M_tilde, xdc_array, states_0, options );

% estimate pulse-echo responses
[ e_B_td, e_B_td_mean, e_B_td_mean_std_dev ] = calibration.estimate_PER_point( u_M_tilde, xdc_array, states_est, options );

%--------------------------------------------------------------------------
% 4.) compute errors
%--------------------------------------------------------------------------
% specify cell arrays
rel_RMSE_positions = cell( size( states_est ) );
rel_RMSE_c_avg = zeros( size( states_est ) );
rel_RMSE_e_B_td = cell( size( states_est ) );
rel_RMSE_e_B_td_mean = cell( size( states_est ) );

% iterate pulse-echo measurement sequences
for index_sequence = 2:numel( sequences )

    positions_est = { states_est{ index_sequence }.position_target };
    positions_est = cat( 1, positions_est{ : } );
    error_pos = positions_est - operators_born( index_sequence ).discretization.spatial.grid_FOV.positions( indices_tpsf, : );
    rel_RMSE_positions{ index_sequence } = vecnorm( error_pos, 2, 2 ) ./ vecnorm( operators_born( index_sequence ).discretization.spatial.grid_FOV.positions( indices_tpsf, : ), 2, 2 );

    error_c_avg = [ states_est{ index_sequence }.c_avg ]' - c_ref( index_sequence );
    rel_RMSE_c_avg( index_sequence ) = norm( error_c_avg ) / ( c_ref( index_sequence ) * sqrt( numel( states_est{ index_sequence } ) ) );

    rel_RMSE_e_B_td{ index_sequence } = cell( size( states_est{ index_sequence } ) );
    rel_RMSE_e_B_td_mean{ index_sequence } = zeros( size( states_est{ index_sequence } ) );

	for index_target = 1:numel( states_est{ index_sequence } )

        % pulse-echo responses
        N_samples_min = min( [ abs( e_B_td{ index_sequence }{ index_target }.axis ), abs( u_tx_tilde.axis ) ] );
        error_e_B_td = e_B_td{ index_sequence }{ index_target }.samples( 1:N_samples_min, : ) / double( operators_born( index_sequence ).discretization.spatial.grid_FOV.cell_ref.volume ) - u_tx_tilde.samples;
        rel_RMSE_e_B_td{ index_sequence }{ index_target } = vecnorm( error_e_B_td, 2, 1 ) / norm( u_tx_tilde.samples );

        error_e_B_td_mean = e_B_td_mean{ index_sequence }{ index_target }.samples( 1:N_samples_min, : ) / double( operators_born( index_sequence ).discretization.spatial.grid_FOV.cell_ref.volume ) - u_tx_tilde.samples;
        rel_RMSE_e_B_td_mean{ index_sequence }( index_target ) = norm( error_e_B_td_mean ) / norm( u_tx_tilde.samples );

%         figure( index_target );
%         plot( (1:N_samples_min), e_B_td{ index_sequence }{ index_target }.samples( 1:N_samples_min, : ) ./ max( e_B_td{ index_sequence }{ index_target }.samples( 1:N_samples_min, : ) ), (1:N_samples_min), u_tx_tilde.samples / max( u_tx_tilde.samples ) )
%         plot( (1:N_samples_min), e_B_td{ index_sequence }{ index_target }.samples( 1:N_samples_min, : ) / double( operators_born( index_sequence ).discretization.spatial.grid_FOV.cell_ref.volume ), (1:N_samples_min), u_tx_tilde.samples, (1:N_samples_min), error_e_B_td )

    end % for index_target = 1:numel( states_est{ index_sequence } )

end % for index_sequence = 1:numel( sequences )
