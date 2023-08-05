% reconstruct point scatterers from simulated RF signals
% material parameter: compressibility
%
% author: Martin Schiffner
% date: 2023-08-01
% modified: 2023-08-04

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;
clc;

% add computational UI toolbox
addpath( genpath( './' ) );

% add SPGL1 (see https://friedlander.io/spgl1)
addpath( genpath( '../external/spgl1/' ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% specify pulse-echo measurement setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 0.) name of the simulation
%--------------------------------------------------------------------------
str_name = 'testbench_3d';

%--------------------------------------------------------------------------
% 1.) transducer array
%--------------------------------------------------------------------------
xdc_array = scattering.sequences.setups.transducers.L14_5_38;

%--------------------------------------------------------------------------
% 2.) homogeneous fluid
%--------------------------------------------------------------------------
% unperturbed mass density
rho_0 = physical_values.kilogram_per_cubicmeter( 1000 );

% time-causal absorption model
c_ref = physical_values.meter_per_second( 1500 );
f_ref = physical_values.hertz( 4e6 );
absorption_model = scattering.sequences.setups.materials.absorption_models.time_causal( 0, 2.17e-3, 2, c_ref, f_ref );

% average group velocity
c_avg = c_ref;

% create homogeneous fluid
homogeneous_fluid = scattering.sequences.setups.materials.homogeneous_fluid( rho_0, absorption_model, c_avg );

%--------------------------------------------------------------------------
% 3.) field of view
%--------------------------------------------------------------------------
FOV_size_lateral = xdc_array.N_elements_axis .* xdc_array.cell_ref.edge_lengths( : );
FOV_size_axial = FOV_size_lateral( 1 );

FOV_offset_axial = 64 * xdc_array.cell_ref.edge_lengths( 1 ) / 4;

FOV_intervals_lateral = num2cell( math.interval( - FOV_size_lateral / 2, FOV_size_lateral / 2 ) );
FOV_interval_axial = math.interval( FOV_offset_axial, FOV_offset_axial + FOV_size_axial );

FOV_cs = scattering.sequences.setups.fields_of_view.orthotope( FOV_intervals_lateral{ : }, FOV_interval_axial );

%--------------------------------------------------------------------------
% 4.) create pulse-echo measurement setup
%--------------------------------------------------------------------------
setup = scattering.sequences.setups.setup( xdc_array, homogeneous_fluid, FOV_cs, str_name );

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
% a) steered QPW
%--------------------------------------------------------------------------
theta_incident = deg2rad( 70 );
e_theta = math.unit_vector( [ cos( theta_incident ), 0, sin( theta_incident ) ] );

% create waves
QPW = scattering.sequences.syntheses.deterministic.qpw( e_theta );

%--------------------------------------------------------------------------
% 2.) receive parameters
%--------------------------------------------------------------------------
% create mixer settings [ read out each element individually ]
controls_rx_identity = scattering.sequences.settings.controls.rx( num2cell( (1:xdc_array.N_elements) ), repmat( num2cell( processing.delta_matrix( 0, T_s ) ), [ 1, xdc_array.N_elements ] ), [], interval_f );

%--------------------------------------------------------------------------
% 3.) create pulse-echo measurement sequences
%--------------------------------------------------------------------------
% create pulse-echo measurement sequences
sequence = scattering.sequences.sequence( setup, u_tx_tilde, processing.delta_matrix( zeros( xdc_array.N_elements, 1 ), T_s ), QPW, controls_rx_identity );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% create scattering operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) specify scattering operator options
%--------------------------------------------------------------------------
% spatial discretization options
method_faces = scattering.sequences.setups.discretizations.methods.grid_numbers( [ 4; 53 ] );
method_FOV = scattering.sequences.setups.discretizations.methods.grid_distances( physical_values.meter( [ 76.2e-6; 4e-3; 76.2e-6 ] ) );
options_disc_spatial = scattering.sequences.setups.discretizations.options( method_faces, method_FOV );

% spectral discretization options
options_disc_spectral = scattering.sequences.settings.discretizations.sequence;

% create discretization options
options_disc = scattering.options.discretization( options_disc_spatial, options_disc_spectral );

% create static scattering operator options
options_static = scattering.options.static( options_disc );

% create momentary scattering operator options
options_momentary = scattering.options.momentary( scattering.options.gpu_off );

% scattering options
options = scattering.options( options_static, options_momentary );

%--------------------------------------------------------------------------
% 2.) create scattering operator [ this may take some time... ]
%--------------------------------------------------------------------------
% create scattering operator (Born approximation)
operator_born = scattering.operator_born( sequence, options );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% received energies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) compute received energies
%--------------------------------------------------------------------------
% select type of time gain compensation (TGC)
tgc = regularization.tgc.off;

% select dictionary for sparse representation
dictionary = regularization.dictionaries.identity;

% create energy options
options_energy = regularization.options.energy_rx( options_momentary, tgc, dictionary );

% call function
E_M = energy_rx( operator_born, options_energy );

%--------------------------------------------------------------------------
% 2.) display result
%--------------------------------------------------------------------------
figure( 1 );
axes_position = get_axes( operator_born.sequence.setup.FOV.shape.grid );
positions_x = double( axes_position( 1 ).members );
positions_z = double( axes_position( 3 ).members );
imagesc( positions_x * 1e3, positions_z * 1e3, illustration.dB( squeeze( reshape( double( E_M ), operator_born.sequence.setup.FOV.shape.grid.N_points_axis ) ).', 20 ), [ -60, 0 ] );
title( 'Received energy' );
xlabel( 'Lateral position (mm)' );
ylabel( 'Axial position (mm)' );
colormap gray;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% point spread functions (PSFs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% select column vectors
indices_tpsf_axis = [ 128, 1, 128; 256, 1, 256; 384, 1, 384 ];
indices_tpsf = forward_index_transform( operator_born.sequence.setup.FOV.shape.grid, indices_tpsf_axis );

% create common regularization options
options_common = regularization.options.common( options_energy, regularization.normalizations.threshold( eps ) );

% create TPSF options
options_tpsf = regularization.options.tpsf( options_common, indices_tpsf );

%--------------------------------------------------------------------------
% 2.) compute PSF
%--------------------------------------------------------------------------
% a) GPU processing
[ gamma_kappa_tpsf, theta_kappa_tpsf, E_M_act, adjointness ] = tpsf( operator_born, options_tpsf );

% b) CPU processing
% [ gamma_kappa_tpsf_cpu, theta_kappa_tpsf_cpu, E_M_cpu, adjointness_cpu ] = tpsf( operators_born, options_tpsf_cpu );

%--------------------------------------------------------------------------
% 2.) display results
%--------------------------------------------------------------------------
figure( 2 );
% iterate TPSFs
for index_tpsf = 1:numel( indices_tpsf )

    % display results
    subplot( 1, numel( indices_tpsf ), index_tpsf );
    imagesc( positions_x * 1e3, positions_z * 1e3, illustration.dB( squeeze( reshape( theta_kappa_tpsf( :, index_tpsf ), operator_born.sequence.setup.FOV.shape.grid.N_points_axis ) )', 20 ), [ -60, 0 ] );
    xlabel( 'Lateral position (mm)' );
    ylabel( 'Axial position (mm)' );
    colormap gray;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simulate pulse-echo measurement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SNR_dB = 3;

%--------------------------------------------------------------------------
% 1.) create wire phantom
%--------------------------------------------------------------------------
gamma_kappa = zeros( operator_born.sequence.size( 2 ), 1 );
gamma_kappa( 127*512+128 ) = 1;
gamma_kappa( 127*512+256 ) = 1;
gamma_kappa( 127*512+384 ) = 1;
gamma_kappa( 255*512+128 ) = 1;
gamma_kappa( 255*512+256 ) = 1;
gamma_kappa( 255*512+384 ) = 1;
gamma_kappa( 383*512+128 ) = 1;
gamma_kappa( 383*512+256 ) = 1;
gamma_kappa( 383*512+384 ) = 1;

%--------------------------------------------------------------------------
% 2.) compute mixed voltage signals
%--------------------------------------------------------------------------
u_M = forward( operator_born, gamma_kappa );
u_M_tilde = signal( u_M );

%--------------------------------------------------------------------------
% 3.) add measurement noise
%--------------------------------------------------------------------------
% TODO: noise power

%--------------------------------------------------------------------------
% 3.) display results
%--------------------------------------------------------------------------
figure( 3 );
imagesc( illustration.dB( abs( hilbert( double( u_M_tilde.samples ) ) ), 20 ), [ -60, 0 ] );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% apply adjoint normalized sensing matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) adjoint scattering
%--------------------------------------------------------------------------
% create common regularization options
options_common = regularization.options.common( options_energy, regularization.normalizations.threshold( eps ) );

% call function
[ gamma_hat, theta_hat, rel_RMSE ] = adjoint( operator_born, u_M, options_common );

%--------------------------------------------------------------------------
% 2.) display results
%--------------------------------------------------------------------------
figure( 4 );
imagesc( positions_x * 1e3, positions_z * 1e3, illustration.dB( squeeze( reshape( theta_hat, operator_born.sequence.setup.FOV.shape.grid.N_points_axis ) )', 20 ), [ -60, 0 ] );
xlabel( 'Lateral position (mm)' );
ylabel( 'Axial position (mm)' );
title( 'Adjoint scattering' );
colormap gray;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% lq-minimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 1.) create lq-minimization options
%--------------------------------------------------------------------------
N_iterations_max = 1e1; % maximum number of iterations (1)
rel_RMSE_max = 1e-1;	% desired relative RMSE (1)

% create l0-minimization options
options_lq = [ regularization.options.lq_minimization( options_common, regularization.algorithms.greedy.omp( rel_RMSE_max, N_iterations_max ) ); ...      % l0-minimization
               regularization.options.lq_minimization( options_common, regularization.algorithms.convex.spgl1( rel_RMSE_max, N_iterations_max, 1 ) ); ... % l1-minimization
               regularization.options.lq_minimization( options_common, regularization.algorithms.convex.spgl1( rel_RMSE_max, N_iterations_max, 2 ) ); ];  % l2-minimization

%--------------------------------------------------------------------------
% 2.) perform lq-minimization
%--------------------------------------------------------------------------
% call function
[ gamma_recon_lq, theta_recon_lq_normed, u_M_res, info ] = lq_minimization( operator_born, u_M, options_lq );

% compute errors
rel_RMSE = zeros( size( options_lq ) );
for index_options = 1:numel( options_lq )
    rel_RMSE( index_options ) = norm( gamma_kappa - gamma_recon_lq( index_options ).samples ) / norm( gamma_kappa );
end

%--------------------------------------------------------------------------
% 3.) display results
%--------------------------------------------------------------------------
figure( 5 );
for index_options = 1:numel( options_lq )

    % show compressibility fluctuations
    subplot( 2, numel( options_lq ), index_options );
    imagesc( positions_x * 1e3, positions_z * 1e3, abs( squeeze( reshape( gamma_recon_lq( index_options ).samples, operator_born.sequence.setup.FOV.shape.grid.N_points_axis ) )' ) );
    xlabel( 'Lateral position (mm)' );
    ylabel( 'Axial position (mm)' );
    title( { 'lq-Minimization', sprintf( 'algorithm: %s', options_lq( index_options ).algorithm ), sprintf( 'rel. RMSE: %.2f %%', rel_RMSE( index_options ) * 1e2 ) } );
    colormap gray;

    % show residual RF data
    u_M_res_tilde = signal( u_M_res{ index_options } );
    subplot( 2, numel( options_lq ), index_options +  numel( options_lq ) );
%     imagesc( abs( hilbert( double( u_M_res_tilde.samples ) ) ) );
    imagesc( double( u_M_res_tilde.samples ) / max( abs( double( u_M_tilde.samples( : ) ) ) ), [ -1, 1 ] );
    title( 'Residual RF data' );

end
