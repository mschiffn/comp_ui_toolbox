% testbench for three-dimensional space
% material parameter: compressibility
%
% author: Martin F. Schiffner
% date: 2019-01-10
% modified: 2019-11-04

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
c_ref = physical_values.meter_per_second( (1460:10:1520) );
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
    % sequence_PW = scattering.sequences.sequence_QPW( setup, repmat( u_tx_tilde, [3,1] ), e_theta( [1, 6, 11] ), interval_t, interval_f );
    % sequence_QSW = scattering.sequences.sequence_QSW( setup, repmat( u_tx_tilde, size( positions_src, 1 ) ), positions_src, angles, data_RF_qsw.interval_f );
    sequences{ index_sequence } = scattering.sequences.sequence_QPW( setup( index_sequence ), repmat( u_tx_tilde, size( e_theta ) ), e_theta, interval_f );

    % sequence = scattering.sequences.sequence_QPW( setup, u_tx_tilde, e_theta( 1 ), interval_f );
    %         sequence = scattering.sequences.sequence_SA( setup, excitation_voltages_common, pi / 2 * ones( 128, 1 ) );
    %         settings_rng_apo = auxiliary.setting_rng( 10 * ones(11, 1), repmat({'twister'}, [ 11, 1 ]) );
    %         settings_rng_del = auxiliary.setting_rng( 3 * ones(1, 1), repmat({'twister'}, [ 1, 1 ]) );
    %         sequence = scattering.sequences.sequence_rnd_apo( setup, excitation_voltages_common, settings_rng_apo );
    %         e_dir = math.unit_vector( [ cos( 89.9 * pi / 180 ), sin( 89.9 * pi / 180 ) ] );
    %         sequence = scattering.sequences.sequence_rnd_del( setup, excitation_voltages_common, e_dir, settings_rng_del );
    %         sequence = scattering.sequences.sequence_rnd_apo_del( setup, excitation_voltages_common, e_dir, settings_rng_apo, settings_rng_del );

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initialize scattering operators (Born approximation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TODO: turn off anti-aliasing in p_in!

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
options_momentary = scattering.options.momentary( scattering.options.anti_aliasing_off );

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
u_M = forward( operators_born, theta, [], scattering.options.gpu_off, scattering.options.anti_aliasing_off );
u_M = cat( 2, u_M{:} );

lbs_q = zeros( size( u_M ) );
for index_sequence = 1:numel( sequences )

	temp = quantize( operators_born( index_sequence ).sequence.interval_hull_t, T_s );
    lbs_q( index_sequence ) = temp.q_lb;
end

u_M_tilde = signal( u_M, lbs_q, T_s );

%--------------------------------------------------------------------------
% display results
%--------------------------------------------------------------------------
figure( 1 );
imagesc( illustration.dB( abs( hilbert( double( u_M_tilde( 1 ).samples ) ) ), 20 ), [ -60, 0 ] );

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

c_avg_start = c_ref;%physical_values.meter_per_second( 1490 * ones( size( sequences ) ) );

% specify cell array for states_0
states_0 = cell( size( sequences ) );

% iterate pulse-echo measurement sequences
for index_sequence = 1:numel( sequences )

	% create initial states
% 	pos_start = [ xdc_array.positions_ctr( indices_elements{ index_sequence }, : ), u_M_tilde( index_sequence ).axis.members( indices_t{ index_sequence }( : ) ) * c_avg_start( index_sequence ) / 2 ];
    pos_start = operators_born( index_sequence ).sequence.setup.FOV.shape.grid.positions( indices_tpsf, : );
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
% estimate speed of sound
[ states_est, rel_RMSE, pulse_shape_mean, pulse_shape_std_dev ] = calibration.estimate_SOS_point( u_M_tilde, xdc_array, states_0, options );

% estimate pulse-echo responses
[ e_B_tilde, e_B_tilde_mean, e_B_tilde_std_dev ] = calibration.estimate_PER_point( u_M_tilde( 1 ), xdc_array, states_0{1}, options );

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
	rel_RMSE_c_avg{ index_sequence } = vecnorm( error_c_avg, 2, 2 ) / c_ref( index_sequence );

	% mean SoS
	weights = 1 ./ rel_RMSE{ index_sequence };
	weights = weights / sum( weights );
	c_avg_mean = [ states_est{ index_sequence }.c_avg ] * weights;
	rel_RMSE_c_avg_mean( index_sequence ) = abs( c_avg_mean - c_ref( index_sequence ) ) / c_ref( index_sequence );

    rel_RMSE_e_B_tilde{ index_sequence } = cell( size( states_est{ index_sequence } ) );
    rel_RMSE_e_B_tilde_mean{ index_sequence } = zeros( size( states_est{ index_sequence } ) );

	for index_target = 1:4%numel( states_est{ index_sequence } )

        % pulse-echo responses
%         N_samples_min = min( [ abs( e_B_tilde{ index_sequence }{ index_target }.axis ), abs( u_tx_tilde.axis ) ] );
%         error_e_B_td = e_B_tilde{ index_sequence }{ index_target }.samples( 1:N_samples_min, : ) / double( operators_born( index_sequence ).sequence.setup.FOV.shape.grid.cell_ref.volume ) - u_tx_tilde.samples;
%         rel_RMSE_e_B_tilde{ index_sequence }{ index_target } = vecnorm( error_e_B_td, 2, 1 ) / norm( u_tx_tilde.samples );

%         error_e_B_td_mean = e_B_tilde_mean{ index_sequence }{ index_target }.samples( 1:N_samples_min, : ) / double( operators_born( index_sequence ).sequence.setup.FOV.shape.grid.cell_ref.volume ) - u_tx_tilde.samples;
%         rel_RMSE_e_B_tilde_mean{ index_sequence }( index_target ) = norm( error_e_B_td_mean ) / norm( u_tx_tilde.samples );

        figure( index_target );
        plot( e_B_tilde{index_target}.axis.members, e_B_tilde{index_target}.samples ./ max( e_B_tilde{index_target}.samples, [], 1 ), u_tx_tilde.axis.members, u_tx_tilde.samples / max( u_tx_tilde.samples ) );
%         plot( e_B_tilde{ index_sequence }{ index_target }.axis.members, e_B_tilde{ index_sequence }{ index_target }.samples ./ max( e_B_tilde{ index_sequence }{ index_target }.samples, [], 1 ), u_tx_tilde.axis.members, u_tx_tilde.samples / max( u_tx_tilde.samples ) )
%         plot( (1:N_samples_min), e_B_tilde{ index_sequence }{ index_target }.samples( 1:N_samples_min, : ) / double( operators_born( index_sequence ).sequence.setup.FOV.shape.grid.cell_ref.volume ), (1:N_samples_min), u_tx_tilde.samples, (1:N_samples_min), error_e_B_td )

    end % for index_target = 1:numel( states_est{ index_sequence } )

end % for index_sequence = 1:numel( sequences )
