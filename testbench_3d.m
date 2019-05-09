% reconstruct object A from simulated RF signals
% material parameter: compressibility
%
% author: Martin Schiffner
% date: 2019-01-10

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;
clc;

addpath( genpath( './' ) );
addpath( genpath( sprintf('%s/toolbox/gpu_bf_toolbox/', matlabroot) ) );
addpath( genpath( sprintf('%s/toolbox/spgl1-1.8/', matlabroot) ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% physical parameters of linear array L14-5/38
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xdc_array = transducers.L14_5_38( 2 );
% xdc_array = transducers.array_planar( transducers.parameters_test, 2 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% general parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% name of the simulation
str_name = 'wire_phantom';

% signal processing parameters
T_s = physical_values.second( 1 / 20e6 );

% specify bandwidth to perform simulation in
f_tx = physical_values.hertz( 4e6 );
frac_bw = 0.7;                  % fractional bandwidth of incident pulse
frac_bw_ref = -60;              % dB value that determines frac_bw

% properties of the homogeneous fluid
% TODO: must match c_avg in setup?
c_ref = physical_values.meter_per_second( 1500 );
absorption_model = absorption_models.time_causal( 0, 0.5, 1, c_ref, f_tx, 1 );

% directions of incidence
theta_incident = (77.5:2.5:102.5) * pi / 180;
e_theta = math.unit_vector( [ cos( theta_incident(:) ), zeros( numel( theta_incident ), 1 ), sin( theta_incident(:) ) ] );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% define field of view
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FOV_size_lateral = xdc_array.parameters.N_elements_axis .* xdc_array.element_pitch_axis;
FOV_size_axial = FOV_size_lateral( 1 );

FOV_intervals_lateral = num2cell( math.interval( - FOV_size_lateral ./ 2, FOV_size_lateral ./ 2 ) );
FOV_interval_axial = math.interval( physical_values.meter( 0 ), FOV_size_axial );

FOV_cs = fields_of_view.orthotope( FOV_intervals_lateral{ : }, FOV_interval_axial );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% reconstruct material parameters with CS ((quasi) plane waves)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify recording time interval
t_lb = 0 .* T_s;        % lower cut-off time
t_ub = 1700 .* T_s;     % upper cut-off time
interval_t = math.interval( t_lb, t_ub );

% specify bandwidth to perform simulation in
f_lb = f_tx .* ( 1 - 0.5 * frac_bw );       % lower cut-off frequency
f_ub = f_tx .* ( 1 + 0.5 * frac_bw );       % upper cut-off frequency
interval_f = math.interval( f_lb, f_ub );

% create pulse-echo measurement setup
setup = pulse_echo_measurements.setup( xdc_array, FOV_cs, absorption_model, str_name );

% specify common excitation voltages
tc = gauspuls( 'cutoff', double( f_tx ), frac_bw, frac_bw_ref, -60 );     % calculate cutoff time
t = (-tc:double(T_s):tc);
pulse = gauspuls( t, double( f_tx ), frac_bw, frac_bw_ref );
axis_t = math.sequence_increasing_regular( 0, numel( t ) - 1, T_s );
u_tx_tilde = discretizations.signal( axis_t, physical_values.voltage( pulse ) );

% create pulse-echo measurement sequence
sequence = pulse_echo_measurements.sequence_QPW( setup, u_tx_tilde, e_theta( 4 ), interval_t, interval_f );
%         sequence = pulse_echo_measurements.sequence_SA( setup, excitation_voltages_common, pi / 2 * ones( 128, 1 ) );
%         settings_rng_apo = auxiliary.setting_rng( 10 * ones(11, 1), repmat({'twister'}, [ 11, 1 ]) );
%         settings_rng_del = auxiliary.setting_rng( 3 * ones(1, 1), repmat({'twister'}, [ 1, 1 ]) );
%         sequence = pulse_echo_measurements.sequence_rnd_apo( setup, excitation_voltages_common, settings_rng_apo );
%         e_dir = math.unit_vector( [ cos( 89.9 * pi / 180 ), sin( 89.9 * pi / 180 ) ] );
%         sequence = pulse_echo_measurements.sequence_rnd_del( setup, excitation_voltages_common, e_dir, settings_rng_del );
%         sequence = pulse_echo_measurements.sequence_rnd_apo_del( setup, excitation_voltages_common, e_dir, settings_rng_apo, settings_rng_del );

%--------------------------------------------------------------------------
% specify options
%--------------------------------------------------------------------------
% discretization options
parameters_elements = discretizations.parameters_number( [ 2, 4 ] );
% parameters_elements = discretizations.parameters_number( [ 4, 53 ] );
parameters_FOV = discretizations.parameters_distance( physical_values.meter( [ 76.2e-6, 4e-3, 76.2e-6 ] ) );
options_disc_spatial = discretizations.options_spatial_grid( parameters_FOV, parameters_elements );
options_disc_spectral = discretizations.options_spectral.signal;
options_disc = discretizations.options( options_disc_spatial, options_disc_spectral );

% scattering options
options = scattering.options( options_disc );

%--------------------------------------------------------------------------
% initialize scattering operator (Born approximation)
%--------------------------------------------------------------------------
operator_born = scattering.operator_born( sequence, options );

%--------------------------------------------------------------------------
% test scattering operator
%--------------------------------------------------------------------------
% specify coefficient vector
theta = zeros( 512^2, 1 );
% indices = randperm( 512^2 );
% indices = indices(1:10);
% theta( indices ) = 1;
theta(383*512+128) = 1;
theta(255*512+256) = 1;
theta(383*512+384) = 1;
% 
% theta(512^2+383*512+64) = 2;
% theta(512^2+255*512+156) = 2;
% theta(512^2+383*512+284) = 2;

% define linear transforms
% TODO: enumerate wavelet names / install wavelet toolbox
% LT_d20 = linear_transforms.wavelet( 'daubechies', 20, 512, 0 );
LT_weighting = linear_transforms.weighting( 1 ./ sqrt( double( operator_born.E_rx ) ) );
LT_fourier_blk = linear_transforms.fourier_block( operator_born.discretization.spatial.grid_FOV.N_points_axis([1,3]), operator_born.discretization.spatial.grid_FOV.N_points_axis([1,3]) / 16 );

% perform forward scattering
profile on
u_rx = forward_quick( operator_born, theta, LT_weighting );
u_rx = forward( operator_born, theta );
profile viewer

% perform adjoint scattering
theta_hat = adjoint_quick( operator_born, u_rx );
theta_hat_weighting = adjoint_quick( operator_born, u_rx, LT_weighting );

% transform point spread functions
[ theta_hat_tpsf, E_rx, adjointness ] = tpsf( operator_born, [ 383*512+128 ], LT_weighting );

u_rx_tilde = signal( u_rx, 0, T_s );

%--------------------------------------------------------------------------
% display results
%--------------------------------------------------------------------------
figure( 1 );
subplot(1,2,1);
imagesc( double( u_rx_tilde.samples )' );
subplot(1,2,2);
imagesc( illustration.dB( abs( hilbert( double( u_rx_tilde.samples )' ) ), 20 ), [ -60, 0 ] );



