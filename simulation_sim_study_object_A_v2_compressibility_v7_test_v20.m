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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% general parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% name of the simulation
str_name = 'wire_phantom';

% signal processing parameters
T_s = physical_values.second( 1 / 20e6 );

% specify bandwidth to perform simulation in
f_tx = physical_values.hertz( 4e6 );
frac_bw = 1;                   % fractional bandwidth of incident pulse
frac_bw_ref = -60;             % dB value that determines frac_bw

% properties of the homogeneous fluid
c_ref = physical_values.meter_per_second( 1500 );
absorption_model = absorption_models.time_causal( 0, 2.17e-3, 2, double( c_ref ), double( f_tx ), 1 );

% directions of incidence
theta_incident = (77.5:2.5:102.5) * pi / 180;
e_theta = math.unit_vector( [ cos( theta_incident(:) ), zeros( numel( theta_incident ), 1 ), sin( theta_incident(:) ) ] );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% define field of view
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FOV_size_axis = xdc_array.parameters.N_elements_axis(1) * xdc_array.element_pitch_axis(1) * ones( 1, 3 );
FOV_size_axis( 2 ) = physical_values.meter( 76.5e-6 );

FOV_intervals_lateral = math.interval( - FOV_size_axis( 1:2 ) ./ 2, FOV_size_axis( 1:2 ) ./ 2 );
FOV_interval_axial = math.interval( physical_values.meter( 0 ), FOV_size_axis( 1 ) );

FOV_cs = fields_of_view.orthotope( FOV_intervals_lateral(1), FOV_intervals_lateral(2), FOV_interval_axial );

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
sequence = pulse_echo_measurements.sequence_QPW( setup, u_tx_tilde, e_theta( 1 ), interval_t, interval_f );
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
parameters_FOV = discretizations.parameters_distance( physical_values.meter( 76.2e-6 * ones(1, 3) ) );
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
theta = zeros(512^2, 1);
theta(128*512+128) = 1;
theta(256*512+256) = 1;
theta(384*512+384) = 1;

u_rx = operator_born * theta;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute tranform point spread function (TPSF, (quasi) plane waves)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dynamic_range_dB = 70;

% define options for TPSF computation
cs_2d_mlfma_options.mode = 'tpsf';
cs_2d_mlfma_options.material_parameter = 1;
cs_2d_mlfma_options.gpu_index = 0;
cs_2d_mlfma_options.excitation = 'cylindrical_waves';
cs_2d_mlfma_options.normalize_columns_threshold = 0;    % deactivate thresholding

% choose TPSF indices
N_coordinates = 9;
direction = N_lattice_axis_cs' - ones(2,1);
tpsf_coordinates = round( ones(N_coordinates, 2) + linspace(0.05, 0.95, N_coordinates)' * direction' );
cs_2d_mlfma_options.tpsf_indices = [tpsf_coordinates(:,1) - 1, tpsf_coordinates(:,2)] * [N_lattice_axis_cs(2); 1];
N_tpsf = numel(cs_2d_mlfma_options.tpsf_indices);

% create info string
str_tpsf_params = sprintf('%s_%d_%d_%.2f_%.2f_exc_%s_theta%s_f_lb_%.2f_f_ub_%.2f_trans%s_abs_%s_thresh_%.2f', str_name, N_lattice_axis_cs(1), N_lattice_axis_cs(2), lattice_delta_x_cs * 1e4, lattice_delta_z_cs * 1e4, cs_2d_mlfma_options.excitation, str_theta_cs, f_lb_cs / 1e6, f_ub_cs / 1e6, str_transform, str_absorption, cs_2d_mlfma_options.normalize_columns_threshold * 1e2);

% compute TPSFs
[theta_tpsf, gamma_tpsf, column_norms, adjointness] = forward_simulation_mlfma_gpu_v18(zeros(N_lattice_axis_cs(2),N_lattice_axis_cs(1)), zeros(N_lattice_axis_cs(2),N_lattice_axis_cs(1)), A_in_td, f_lb_cs, f_ub_cs, N_elements, element_width, element_kerf, factor_interp_cs, lattice_delta_z_cs, lattice_pos_z_shift_cs, theta_incident(indices_theta_recon(indices_theta_cs)), c_ref, f_s, cs_2d_mlfma_options);

% format and save results
if cs_2d_mlfma_options.material_parameter == 0
    
    % both material parameters
    
    % allocate memory
    theta_kappa_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    theta_rho_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    gamma_kappa_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    gamma_rho_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    
    % separate and plot TPSFs
    for index_tpsf = 1:N_tpsf
        
        % both material parameters  
        theta_kappa_tpsf(:, :, index_tpsf) = theta_tpsf(:, 1:N_lattice_axis_cs(1), index_tpsf);
        theta_rho_tpsf(:, :, index_tpsf) = theta_tpsf(:, (N_lattice_axis_cs(1) + 1):end, index_tpsf);
        
        gamma_kappa_tpsf(:, :, index_tpsf) = gamma_tpsf(:, 1:N_lattice_axis_cs(1), index_tpsf);
        gamma_rho_tpsf(:, :, index_tpsf) = gamma_tpsf(:, (N_lattice_axis_cs(1) + 1):end, index_tpsf);
        
        % logarithmic compression
        theta_kappa_tpsf_dB = 20*log10(abs(theta_kappa_tpsf(:, :, index_tpsf)) / max(max(abs(theta_kappa_tpsf(:, :, index_tpsf)))));
        theta_rho_tpsf_dB = 20*log10(abs(theta_rho_tpsf(:, :, index_tpsf)) / max(max(abs(theta_rho_tpsf(:, :, index_tpsf)))));
        
        gamma_kappa_tpsf_dB = 20*log10(abs(gamma_kappa_tpsf(:, :, index_tpsf)) / max(max(abs(gamma_kappa_tpsf(:, :, index_tpsf)))));
        gamma_rho_tpsf_dB = 20*log10(abs(gamma_rho_tpsf(:, :, index_tpsf)) / max(max(abs(gamma_rho_tpsf(:, :, index_tpsf)))));
    
        % difference tpsf
        gamma_tpsf_diff = gamma_kappa_tpsf(:, :, index_tpsf) - gamma_rho_tpsf(:, :, index_tpsf);
        gamma_tpsf_diff_dB = 20*log10(abs(gamma_tpsf_diff) / max(abs(gamma_tpsf_diff(:))));

        % indices of tpsf
        index_tpsf_x = ceil(cs_2d_mlfma_options.tpsf_indices(index_tpsf) / N_lattice_axis_cs(2));
        index_tpsf_z = cs_2d_mlfma_options.tpsf_indices(index_tpsf) - (index_tpsf_x - 1) * N_lattice_axis_cs(2);

        % display results
        figure(index_tpsf);
        % results for kappa
        subplot(3,3,1);
        imagesc(theta_kappa_tpsf_dB, [-dynamic_range_dB, 0]);
        if index_tpsf_x <= N_lattice_axis_cs(1);
            line([index_tpsf_x - 2, index_tpsf_x + 2], index_tpsf_z * ones(1,2), 'Color', 'r');
            line(index_tpsf_x * ones(1,2), [index_tpsf_z - 2, index_tpsf_z + 2], 'Color', 'r');
        end
        subplot(3,3,2);
        imagesc(lattice_pos_x_cs * 1e3, lattice_pos_z_cs * 1e3, gamma_kappa_tpsf_dB, [-dynamic_range_dB, 0]);
        subplot(3,3,3);
        gamma_kappa_tpsf_dft = fft2(gamma_kappa_tpsf(:, :, index_tpsf));
        gamma_kappa_tpsf_dft_dB = 20*log10(abs(gamma_kappa_tpsf_dft) / max(abs(gamma_kappa_tpsf_dft(:))));
        imagesc(axis_k_hat_x_cs, axis_k_hat_z_cs, fftshift(gamma_kappa_tpsf_dft_dB, 2), [-dynamic_range_dB, 0]);
        draw_FDT_pw(2*pi*f_lb_cs / c_ref, 2*pi*f_ub_cs / c_ref, theta_incident(indices_theta_recon(indices_theta_cs)), []);
        % results for rho
        subplot(3,3,4);
        imagesc(theta_rho_tpsf_dB, [-dynamic_range_dB, 0]);
        if index_tpsf_x > N_lattice_axis_cs(1);
            line([index_tpsf_x - 2, index_tpsf_x + 2] - N_lattice_axis_cs(1), index_tpsf_z * ones(1,2), 'Color', 'r');
            line(index_tpsf_x * ones(1,2) - N_lattice_axis_cs(1), [index_tpsf_z - 2, index_tpsf_z + 2], 'Color', 'r');
        end
        subplot(3,3,5);
        imagesc(lattice_pos_x_cs * 1e3, lattice_pos_z_cs * 1e3, gamma_rho_tpsf_dB, [-dynamic_range_dB, 0]);
        subplot(3,3,6);
        gamma_rho_tpsf_dft = fft2(gamma_rho_tpsf(:, :, index_tpsf));
        gamma_rho_tpsf_dft_dB = 20*log10(abs(gamma_rho_tpsf_dft) / max(abs(gamma_rho_tpsf_dft(:))));
        imagesc(axis_k_hat_x_cs, axis_k_hat_z_cs, fftshift(gamma_rho_tpsf_dft_dB, 2), [-dynamic_range_dB, 0]);
        draw_FDT_pw(2*pi*f_lb_cs / c_ref, 2*pi*f_ub_cs / c_ref, theta_incident(indices_theta_recon(indices_theta_cs)), []);
        subplot(3,3,7);
        imagesc(gamma_tpsf_diff_dB, [-dynamic_range_dB, 0]);
        colormap gray;
    end
        
    % save data as mat file
    str_filename = sprintf('%s/2d_tpsf_kappa_rho_%s.mat', str_path, str_tpsf_params);
    save(str_filename, 'theta_kappa_tpsf', 'gamma_kappa_tpsf', 'theta_rho_tpsf', 'gamma_rho_tpsf', 'N_tpsf', 'lattice_pos_x_cs', 'lattice_pos_z_cs', 'N_lattice_axis_cs', 'lattice_delta_x_cs', 'lattice_delta_z_cs', 'axis_k_hat_x_cs', 'axis_k_hat_z_cs', 'f_lb_cs', 'f_ub_cs', 'c_ref', 'indices_theta_cs', 'indices_theta_recon', 'theta_incident', 'cs_2d_mlfma_options', 'column_norms', 'adjointness');

elseif cs_2d_mlfma_options.material_parameter == 1
        
    % only gamma_kappa
    
    % allocate memory
    theta_kappa_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    gamma_kappa_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
        
    % plot TPSFs
    for index_tpsf = 1:N_tpsf

        % only gamma_kappa
        theta_kappa_tpsf(:, :, index_tpsf) = theta_tpsf(:, 1:N_lattice_axis_cs(1), index_tpsf); 
        gamma_kappa_tpsf(:, :, index_tpsf) = gamma_tpsf(:, 1:N_lattice_axis_cs(1), index_tpsf);

        % logarithmic compression
        theta_kappa_tpsf_dB = 20*log10(abs(theta_kappa_tpsf(:, :, index_tpsf)) / max(max(abs(theta_kappa_tpsf(:, :, index_tpsf)))));
        gamma_kappa_tpsf_dB = 20*log10(abs(gamma_kappa_tpsf(:, :, index_tpsf)) / max(max(abs(gamma_kappa_tpsf(:, :, index_tpsf)))));

        % indices of tpsf
        index_tpsf_x = ceil(cs_2d_mlfma_options.tpsf_indices(index_tpsf) / N_lattice_axis_cs(2));
        index_tpsf_z = cs_2d_mlfma_options.tpsf_indices(index_tpsf) - (index_tpsf_x - 1) * N_lattice_axis_cs(2);

        % display results
        figure(index_tpsf);
        subplot(1,3,1);
        imagesc(theta_kappa_tpsf_dB, [-dynamic_range_dB, 0]);
        line([index_tpsf_x - 2, index_tpsf_x + 2], index_tpsf_z * ones(1,2), 'Color', 'r');
        line(index_tpsf_x * ones(1,2), [index_tpsf_z - 2, index_tpsf_z + 2], 'Color', 'r');
        subplot(1,3,2);
        imagesc(lattice_pos_x_cs * 1e3, lattice_pos_z_cs * 1e3, gamma_kappa_tpsf_dB, [-dynamic_range_dB, 0]);
        subplot(1,3,3);
        gamma_kappa_tpsf_dft = fft2(gamma_kappa_tpsf(:, :, index_tpsf));
        gamma_kappa_tpsf_dft_dB = 20*log10(abs(gamma_kappa_tpsf_dft) / max(abs(gamma_kappa_tpsf_dft(:))));
        imagesc(axis_k_hat_x_cs, axis_k_hat_z_cs, fftshift(gamma_kappa_tpsf_dft_dB, 2), [-dynamic_range_dB, 0]);
        draw_FDT_pw(2*pi*f_lb_cs / c_ref, 2*pi*f_ub_cs / c_ref, theta_incident(indices_theta_recon(indices_theta_cs)), []);
        colormap gray;
    end
        
    % save data as mat file   
    str_filename = sprintf('%s/2d_tpsf_kappa_%s.mat', str_path, str_tpsf_params);
    save(str_filename, 'theta_kappa_tpsf', 'gamma_kappa_tpsf', 'N_tpsf', 'lattice_pos_x_cs', 'lattice_pos_z_cs', 'N_lattice_axis_cs', 'lattice_delta_x_cs', 'lattice_delta_z_cs', 'axis_k_hat_x_cs', 'axis_k_hat_z_cs', 'f_lb_cs', 'f_ub_cs', 'c_ref', 'indices_theta_cs', 'indices_theta_recon', 'theta_incident', 'cs_2d_mlfma_options', 'column_norms', 'adjointness');

elseif cs_2d_mlfma_options.material_parameter == 2
        
	% only gamma_rho
	
    % allocate memory
    theta_rho_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    gamma_rho_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    
    % plot TPSFs
    for index_tpsf = 1:N_tpsf
        
        % only gamma_rho
        theta_rho_tpsf(:, :, index_tpsf) = theta_tpsf(:, (N_lattice_axis_cs(1) + 1):end, index_tpsf);
        gamma_rho_tpsf(:, :, index_tpsf) = gamma_tpsf(:, (N_lattice_axis_cs(1) + 1):end, index_tpsf);
        
        % logarithmic compression
        theta_rho_tpsf_dB = 20*log10(abs(theta_rho_tpsf(:, :, index_tpsf)) / max(max(abs(theta_rho_tpsf(:, :, index_tpsf)))));
        gamma_rho_tpsf_dB = 20*log10(abs(gamma_rho_tpsf(:, :, index_tpsf)) / max(max(abs(gamma_rho_tpsf(:, :, index_tpsf)))));

        % indices of tpsf
        index_tpsf_x = ceil(cs_2d_mlfma_options.tpsf_indices(index_tpsf) / N_lattice_axis_cs(2));
        index_tpsf_z = cs_2d_mlfma_options.tpsf_indices(index_tpsf) - (index_tpsf_x - 1) * N_lattice_axis_cs(2);

        % display results
        figure(index_tpsf);
        subplot(1,3,1);
        imagesc(theta_rho_tpsf_dB, [-dynamic_range_dB, 0]);
        line([index_tpsf_x - 2, index_tpsf_x + 2], index_tpsf_z * ones(1,2), 'Color', 'r');
        line(index_tpsf_x * ones(1,2), [index_tpsf_z - 2, index_tpsf_z + 2], 'Color', 'r');
        subplot(1,3,2);
        imagesc(lattice_pos_x_cs * 1e3, lattice_pos_z_cs * 1e3, gamma_rho_tpsf_dB, [-dynamic_range_dB, 0]);
        subplot(1,3,3);
        gamma_rho_tpsf_dft = fft2(gamma_rho_tpsf(:, :, index_tpsf));
        gamma_rho_tpsf_dft_dB = 20*log10(abs(gamma_rho_tpsf_dft) / max(abs(gamma_rho_tpsf_dft(:))));
        imagesc(axis_k_hat_x_cs, axis_k_hat_z_cs, fftshift(gamma_rho_tpsf_dft_dB, 2), [-dynamic_range_dB, 0]);
        draw_FDT_pw(2*pi*f_lb_cs / c_ref, 2*pi*f_ub_cs / c_ref, theta_incident(indices_theta_recon(indices_theta_cs)), []);
        colormap gray;
    end
        
	% save data as mat file   
	str_filename = sprintf('%s/2d_tpsf_rho_%s.mat', str_path, str_tpsf_params);
	save(str_filename, 'theta_rho_tpsf', 'gamma_rho_tpsf', 'N_tpsf', 'lattice_pos_x_cs', 'lattice_pos_z_cs', 'N_lattice_axis_cs', 'lattice_delta_x_cs', 'lattice_delta_z_cs', 'axis_k_hat_x_cs', 'axis_k_hat_z_cs', 'f_lb_cs', 'f_ub_cs', 'c_ref', 'indices_theta_cs', 'indices_theta_recon', 'theta_incident', 'cs_2d_mlfma_options', 'column_norms', 'adjointness');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load and process SAFT data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TGC parameters
tgc_sa = false;                        % use TGC
tgc_absorption_sa = 2.17e-3;           % absorption (dB / (MHz^ypsilon * cm))
tgc_absorption_constant_sa = 0;        % constant absorption (dB / cm)
tgc_ypsilon_sa = 2;                    % exponent in power law for absorption

% load SAFT data
str_filename_saft = sprintf('RF_data/simulated/%s/data_RF_%s_exc_cylindrical_waves_saft_20MHz_4.0MHz_512_512_0.76_0.76_f_lb_%.2f_f_ub_%.2f_trans_none_none_abs_power_law_0.00_0.00_2.00.mat', str_name, str_name, f_lb / 1e6, f_ub / 1e6);
data_saft = load( str_filename_saft );

%--------------------------------------------------------------------------
%  create RF data for standard reconstruction algorithms
%--------------------------------------------------------------------------
data_saft.data_RF = zeros(data_saft.N_samples_t, N_elements, N_elements);
for index_element_tx = 1:N_elements
    
    data_saft.data_RF(:, :, index_element_tx) = data_saft.pressure_born_kappa_td{index_element_tx}';
end

%--------------------------------------------------------------------------
% apply artificial TGC
%--------------------------------------------------------------------------
str_TGC_sa = 'off';
data_saft.data_RF_tgc = data_saft.data_RF;
if tgc_sa
    distance_spherical_prop = (1:data_saft.N_samples_t) * c_ref / f_s;   % total propagation distance (m)

    exponent = tgc_absorption_sa * log(10) * f_tx^tgc_ypsilon_sa / (20 * 0.01 * (1e6)^tgc_ypsilon_sa);
    exponent_constant = tgc_absorption_constant_sa * log(10) / (20 * 0.01);
    factor_tgc = exp((exponent + exponent_constant) * distance_spherical_prop);
    factor_tgc = factor_tgc / min(factor_tgc);

    data_saft.data_RF_tgc = data_saft.data_RF .* repmat(factor_tgc', [1, N_elements, N_elements]);

    str_TGC_sa = sprintf('%.2f_%.2f_%.2f', tgc_absorption_constant_sa, tgc_absorption_sa, tgc_ypsilon_sa);
end

%--------------------------------------------------------------------------
% determine signal and noise powers, reference random numbers
%--------------------------------------------------------------------------
% determine signal power using reference data (reference is synthesized plane wave)
data_saft.data_RF_tgc_ref = data_saft.data_RF_tgc(:,:,64);
data_saft.data_RF_tgc_ref_energy = norm( data_saft.data_RF_tgc_ref(:) )^2;
data_saft.data_RF_tgc_ref_power_mean = data_saft.data_RF_tgc_ref_energy / numel( data_saft.data_RF_tgc_ref(:) );

% initialize random number generator
rng(seed_ref, 'twister');

% create seeds for noise (each SNR, each tx element)
seeds_noise = randperm(N_SNR * N_elements);
seeds_noise = reshape(seeds_noise, [N_SNR, N_elements]);

% info string
str_info_saft = sprintf('%s_%d_%d_%.2f_%.2f_f_lb_%.2f_f_ub_%.2f_tgc_%s', str_name, N_lattice_axis_saft(1), N_lattice_axis_saft(2), lattice_delta_x_saft * 1e4, lattice_delta_z_saft * 1e4, f_lb / 1e6, f_ub / 1e6, str_TGC_sa);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% reconstruct material parameters with SAFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify options
dynamic_range_dB = 70;
index_gpu = 0;
index_tx_plot = 64;

% iterate over specified SNR values
for index_SNR = 1:N_SNR

    % SNR string
    str_SNR = func_create_SNR_string( SNR(index_SNR) );
    
    % create measurement noise  
    data_saft.noise_RF_tgc = zeros(data_saft.N_samples_t, N_elements, N_elements);
    
    for index_element_tx = 1:N_elements
    
        % initialize random number generator
        rng( seeds_noise(index_SNR, index_element_tx), 'twister' );
    
        % calculate measurement noise
        data_saft.noise_RF_tgc(:, :, index_element_tx) = sqrt( noise_RF_tgc_ref_variance(index_SNR) ) * randn(data_saft.N_samples_t, N_elements);
    end

	% check generated SNR value
	fprintf('current SNR = %.1f dB (desired: %d dB)\n', 20*log10( norm( data_saft.data_RF_tgc(:) ) / norm(  data_saft.noise_RF_tgc(:) ) ), SNR(index_SNR));
    
	% add measurement noise
	data_saft.data_RF_tgc_noisy = data_saft.data_RF_tgc + data_saft.noise_RF_tgc;
    
    %--------------------------------------------------------------------------
    % compute saft images
    %--------------------------------------------------------------------------
    [image_recon_saft_boxcar, weights] = gpu_bf_saft( data_saft.data_RF_tgc_noisy, lattice_pos_x_saft, lattice_pos_z_saft, pos_elements, zeros(1,128), ones(1,128), pos_elements, zeros(1,128), ones(1,128), f_number, f_lb, f_ub, N_samples_shift, c_ref, f_s, index_gpu, 0 );

    %--------------------------------------------------------------------------
    % compute cylindrical wave images
    %--------------------------------------------------------------------------
    % allocate memory for results
    image_recon_das_cw_boxcar = zeros(N_lattice_axis_saft(2), N_lattice_axis_saft(1), N_elements);

    for index_element_tx = 1:N_elements
    
        temp = zeros(data_saft.N_samples_t, N_elements, 2);
        temp(:, :, 1) = data_saft.data_RF_tgc_noisy(:, :, index_element_tx);
    
        [image_recon_das_cw_boxcar(:, :, index_element_tx), weights] = gpu_bf_saft( temp, lattice_pos_x_saft, lattice_pos_z_saft, [pos_elements(index_element_tx), 0], zeros(1,2), ones(1,2), pos_elements, zeros(1,128), ones(1,128), f_number, f_lb, f_ub, N_samples_shift, c_ref, f_s, index_gpu, 0);
    end

    %--------------------------------------------------------------------------
    % display results
    %--------------------------------------------------------------------------
    % logarithmic compression, spectra
    image_recon_saft_boxcar_dB = 20 * log10(abs(image_recon_saft_boxcar) / max(abs(image_recon_saft_boxcar(:))));
    image_recon_saft_boxcar_dft = fftshift(fft2(image_recon_saft_boxcar), 2);
    image_recon_saft_boxcar_dft_dB = 20 * log10(abs(image_recon_saft_boxcar_dft) / max(abs(image_recon_saft_boxcar_dft(:))));

    image_recon_das_cw_boxcar_dB = 20*log10(abs(image_recon_das_cw_boxcar(:, :, index_tx_plot)) / max(max(abs(image_recon_das_cw_boxcar(:, :, index_tx_plot)))));
    image_recon_das_cw_boxcar_dft = fftshift(fft2(image_recon_das_cw_boxcar(:, :, index_tx_plot)), 2);
    image_recon_das_cw_boxcar_dft_dB = 20 * log10(abs(image_recon_das_cw_boxcar_dft) / max(abs(image_recon_das_cw_boxcar_dft(:))));

    figure(1);
    subplot(1,2,1);
    imagesc(lattice_pos_x_saft * 1e3, lattice_pos_z_saft * 1e3, image_recon_saft_boxcar_dB, [-dynamic_range_dB, 0]);
    colormap gray;
    subplot(1,2,2);
    imagesc(axis_k_hat_x_saft, axis_k_hat_z_saft, image_recon_saft_boxcar_dft_dB, [-dynamic_range_dB, 0]);
    draw_FDT_cw(k_lb, k_ub);
    colormap gray;
    
    figure(4);
    subplot(1,2,1);
    imagesc(lattice_pos_x_saft * 1e3, lattice_pos_z_saft * 1e3, image_recon_das_cw_boxcar_dB, [-dynamic_range_dB, 0]);
    subplot(1,2,2);
    imagesc(axis_k_hat_x_saft, axis_k_hat_z_saft, image_recon_das_cw_boxcar_dft_dB, [-dynamic_range_dB, 0]);
    draw_FDT_cw(k_lb, k_ub);
    colormap gray;
    
    % save data as mat file
    str_filename = sprintf('%s/2d_saft_%s_fnum_%.1f_SNR_%s.mat', str_path, str_info_saft, f_number, str_SNR);
    save(str_filename, 'image_recon_saft_boxcar', 'image_recon_das_cw_boxcar', 'lattice_pos_x_saft', 'lattice_pos_z_saft', 'lattice_delta_x_saft', 'lattice_delta_z_saft', 'axis_k_hat_x_saft', 'axis_k_hat_z_saft', 'N_lattice_axis_saft', 'N_samples_shift', 'c_ref', 'f_number', 'f_lb', 'f_ub', 'f_s');

end % for index_SNR = 1:N_SNR

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% reconstruct material parameters with CS (quasi cylindrical waves)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% general parameters
indices_elements_cs = 64;                      % specify emitting elements (from data_saft.data_RF_filt_window)
SNR_cs = [3, 6, 10, 20, 30, inf];              % specify SNR in dB
f_lb_cs = f_lb;
f_ub_cs = f_ub;
dynamic_range_dB = 60;

N_SNR_cs = numel( SNR_cs );                    % number of SNR values
N_recon_per_SNR_cs = 10 * ones(1, N_SNR_cs);   % number of reconstructions per SNR
N_recon_per_SNR_cs(end) = 1;                   % one reconstruction for noiseless data
                         
% define options for CS reconstruction
cs_2d_mlfma_options = cs_2d_mlfma_set_options;

cs_2d_mlfma_options.material_parameter = 1;
cs_2d_mlfma_options.transform = 'none';

cs_2d_mlfma_options.excitation = 'cylindrical_waves';
cs_2d_mlfma_options.normalize_columns = true;
cs_2d_mlfma_options.window_RF_data = false;
cs_2d_mlfma_options.algorithm = 'omp';
cs_2d_mlfma_options.norm = 'l0';
% cs_2d_mlfma_options.q = 0.5;
% cs_2d_mlfma_options.epsilon_n = 1 ./ (1 + (1:5));
cs_2d_mlfma_options.max_iterations = 1e3;
cs_2d_mlfma_options.svd = false;

cs_2d_mlfma_options.absorption_model = 'power_law';
cs_2d_mlfma_options.absorption = 2.17e-3;
cs_2d_mlfma_options.absorption_constant = 0;
cs_2d_mlfma_options.ypsilon = 2;
cs_2d_mlfma_options.phase_velocity_f_ref = f_tx;

cs_2d_mlfma_options.mlfma_level_coarsest = 6;
cs_2d_mlfma_options.mlfma_level_finest = 6;
cs_2d_mlfma_options.gpu_index = 0;

cs_2d_mlfma_options.name = sprintf('./fast_multipole_method/%s', str_name);
cs_2d_mlfma_options.t_shift_max = pos_elements([1, end]) * cos(theta_incident(indices_theta_recon(end)));
cs_2d_mlfma_options.t_shift_max = abs(cs_2d_mlfma_options.t_shift_max(1)-cs_2d_mlfma_options.t_shift_max(end)) / c_ref;

% create elements string for CS
N_elements_cs = numel(indices_elements_cs);
str_elements_cs = sprintf('_%d', indices_elements_cs);

% define directions of emissions (use pi/2 to avoid time shifts)
theta_incident_cw_cs = ones(1, N_elements_cs) * pi / 2;

% reshape RF data (including TGC), specify apodization
data_RF_tgc_cs = zeros(N_elements, data_saft.N_samples_t, N_elements_cs);

cs_2d_mlfma_options.excitation_apodization = zeros(N_elements_cs, N_elements);

for index_element_cs = 1:N_elements_cs

    data_RF_tgc_cs(:, :, index_element_cs) = data_saft.data_RF_tgc(:, :, indices_elements_cs(index_element_cs))';
    
    cs_2d_mlfma_options.excitation_apodization(index_element_cs, indices_elements_cs(index_element_cs)) = 1;
end

% initialize random number generator
rng(seed_ref, 'twister');

% create seeds for noise (each SNR, each direction of incidence, each recon)
seeds_noise_cs = randperm(N_SNR_cs * N_elements_cs * max(N_recon_per_SNR_cs));
seeds_noise_cs = reshape(seeds_noise_cs, [N_SNR_cs, N_elements_cs, max(N_recon_per_SNR_cs)]);
seeds_noise_cs(:, :, 1) = seeds_noise(:, indices_elements_cs);

% determine signal energy and power
data_RF_tgc_cs_energy = norm( data_RF_tgc_cs(:) )^2;
data_RF_tgc_cs_power_mean = data_RF_tgc_cs_energy / numel( data_RF_tgc_cs(:) );

% determine variances of noise for each SNR (in passband)
noise_RF_tgc_cs_variance = data_RF_tgc_ref_power_mean * 10.^(-SNR_cs / 10);
noise_RF_tgc_cs_variance_bp = 2 * noise_RF_tgc_cs_variance * (f_ub_cs - f_lb_cs) / f_s;

% determine energy of noise for each SNR (in passband)
noise_RF_tgc_cs_energy_bp_expectation = numel( data_RF_tgc_cs(:) ) * noise_RF_tgc_cs_variance_bp;

% corresponding SNR in passband
SNR_BP_cs = 10*log10(data_RF_tgc_cs_power_mean ./ noise_RF_tgc_cs_variance_bp);

% actual SNR
SNR_cs_act = 10*log10(data_RF_tgc_cs_power_mean ./ noise_RF_tgc_cs_variance);

% rel. RMSEs for recovery problems (noise energy / (signal energy + noise energy (filtered))
rel_rmse = sqrt( noise_RF_tgc_cs_energy_bp_expectation ./ (data_RF_tgc_cs_energy + noise_RF_tgc_cs_energy_bp_expectation) );

% thresholds for normalization according to actual SNR
norms_cols_thresh = 10.^(-SNR_cs_act / 20);

% avoid rel. RMSE of null to ensure convergence
indicator = rel_rmse < rel_rmse_min;
rel_rmse(indicator) = rel_rmse_min;

% avoid threshold larger than 1 or of null to avoid numerical errors
indicator = norms_cols_thresh > 1;
norms_cols_thresh(indicator) = 1;
indicator = norms_cols_thresh < norms_cols_thresh_min;
norms_cols_thresh(indicator) = norms_cols_thresh_min;

% create absorption string
str_absorption = func_create_absorption_string( cs_2d_mlfma_options );

% create algorithm and norm string
str_algorithm_norm = func_create_algorithm_norm_string( cs_2d_mlfma_options );

% create transform string
str_transform = func_create_transform_string( cs_2d_mlfma_options );

% iterate over SNR values
for index_SNR_cs = 1:N_SNR_cs

    % SNR string
	str_SNR = func_create_SNR_string( SNR_cs(index_SNR_cs) );

    % rel. RMSE for given SNR
    cs_2d_mlfma_options.rel_mse = rel_rmse(index_SNR_cs);

    % set threshold for normalization according to SNR
    cs_2d_mlfma_options.normalize_columns_threshold = norms_cols_thresh(index_SNR_cs);
    
    % create normalization string
    str_normalize = func_create_normalization_string( cs_2d_mlfma_options );

    % create info string 
    str_cs_params = sprintf('%s_%d_%d_%.2f_%.2f_exc_%s_el%s_f_lb_%.2f_f_ub_%.2f_trans%s_abs_%s_normalize_%s_SNR_%s_tgc_%s_alg_%s', str_name, N_lattice_axis_cs(1), N_lattice_axis_cs(2), lattice_delta_x_cs * 1e4, lattice_delta_z_cs * 1e4, cs_2d_mlfma_options.excitation, str_elements_cs, f_lb_cs / 1e6, f_ub_cs / 1e6, str_transform, str_absorption, str_normalize, str_SNR, str_TGC, str_algorithm_norm);

    % create data structure for reconstruction
    theta_kappa_recon = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_recon_per_SNR_cs(index_SNR_cs));
    gamma_kappa_recon = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_recon_per_SNR_cs(index_SNR_cs));
    y_m_res_energy = cell(1, N_recon_per_SNR_cs(index_SNR_cs));
    algorithm_info = cell(1, N_recon_per_SNR_cs(index_SNR_cs));

    % iterate over realizations
    for index_recon = 1:N_recon_per_SNR_cs(index_SNR_cs)

        % create measurement noise
        noise_RF_tgc_cs = zeros(N_elements, data_saft.N_samples_t, N_elements_cs);
        for index_element_cs = 1:N_elements_cs
        
            % initialize random number generator
            rng( seeds_noise_cs(index_SNR_cs, index_element_cs, index_recon), 'twister' );
        
            % calculate measurement noise
            noise_RF_tgc_cs(:, :, index_element_cs) = sqrt( noise_RF_tgc_cs_variance(index_SNR_cs) ) * randn(data_saft.N_samples_t, N_elements)';
            
            error = noise_RF_tgc_cs(:, :, index_element_cs) - data_saft.noise_RF_tgc(:, :, indices_elements_cs(index_element_cs))';
            norm(error(:))
        end

        % add measurement noise
        data_RF_tgc_cs_noisy = data_RF_tgc_cs + noise_RF_tgc_cs;

        % check generated SNR value
        fprintf('current SNR = %.1f dB (desired: %d dB)\n', 20*log10( norm( data_RF_tgc_cs(:) ) / norm( noise_RF_tgc_cs(:) ) ), SNR_cs(index_SNR_cs));

        % solve optimization problem
        [theta_recon, gamma_recon, y_m_res, y_m_res_energy{index_recon}, gradient, algorithm_info{index_recon}] = cs_2d_mlfma_inverse_scattering_v16(data_RF_tgc_cs_noisy, A_in_td, f_lb_cs, f_ub_cs, N_elements, element_width, element_kerf, factor_interp_cs, N_lattice_axis_cs, lattice_delta_z_cs, lattice_pos_z_shift_cs, theta_incident_cw_cs, c_ref, f_s, cs_2d_mlfma_options);
           
        % reset custom operators for SPGL1
        if strcmp(cs_2d_mlfma_options.algorithm, 'spgl1')
        
            algorithm_info{index_recon}.options.project = [];
            algorithm_info{index_recon}.options.primal_norm = [];
            algorithm_info{index_recon}.options.dual_norm = [];
        end
                  
        theta_kappa_recon(:, :, index_recon) = reshape(theta_recon, [N_lattice_axis_cs(2), N_lattice_axis_cs(1)]);
        gamma_kappa_recon(:, :, index_recon) = reshape(gamma_recon, [N_lattice_axis_cs(2), N_lattice_axis_cs(1)]);
        
        theta_kappa_recon_dB = 20*log10(abs(theta_kappa_recon(:, :, index_recon)) / max(max(abs(theta_kappa_recon(:, :, index_recon)))));
        gamma_kappa_recon_dB = 20*log10(abs(gamma_kappa_recon(:, :, index_recon)) / max(max(abs(gamma_kappa_recon(:, :, index_recon)))));
        
        figure( sum(N_recon_per_SNR_cs(1:(index_SNR_cs-1))) + index_recon );
        subplot(1,3,1);
        imagesc(theta_kappa_recon_dB, [-70,0]);
        subplot(1,3,2);
        imagesc(lattice_pos_x_cs * 1e3, lattice_pos_z_cs * 1e3, gamma_kappa_recon_dB, [-dynamic_range_dB, 0]);
        xlabel('lateral x (mm)');
        ylabel('axial z (mm)');
        subplot(1,3,3);
        gamma_kappa_recon_dft = fftshift( fft2( gamma_kappa_recon(:, :, index_recon) ), 2 );
        gamma_kappa_recon_dft_dB = 20*log10(abs(gamma_kappa_recon_dft) / max(abs(gamma_kappa_recon_dft(:))));
        imagesc(axis_k_hat_x_cs, axis_k_hat_z_cs, gamma_kappa_recon_dft_dB, [-dynamic_range_dB, 0]);
        draw_FDT_cw(2*pi*f_lb_cs / c_ref, 2*pi*f_ub_cs / c_ref);
        colormap gray;

        % save data as mat file
        str_filename = sprintf('%s/2d_cs_kappa_%s_rel_mse_%.1f.mat', str_path, str_cs_params, cs_2d_mlfma_options.rel_mse * 1e2);
        save(str_filename, 'theta_kappa_recon', 'gamma_kappa_recon', 'lattice_pos_x_cs', 'lattice_delta_x_cs', 'lattice_pos_z_cs', 'lattice_delta_z_cs', 'N_lattice_axis_cs', 'axis_k_hat_x_cs', 'axis_k_hat_z_cs', 'f_lb_cs', 'f_ub_cs', 'c_ref', 'y_m_res_energy', 'indices_elements_cs', 'indices_theta_recon', 'theta_incident', 'algorithm_info', 'cs_2d_mlfma_options');

    end % for index_recon = 1:N_recon_per_SNR_cs

end % for index_SNR_cs = 1:N_SNR_cs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute tranform point spread functions (TPSF, single cylindrical waves)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dynamic_range_dB = 70;

% define options for TPSF computation
cs_2d_mlfma_options.mode = 'tpsf';
cs_2d_mlfma_options.normalize_columns_threshold = 0;    % deactivate thresholding

% choose TPSF indices
N_coordinates = 9;
direction = N_lattice_axis_cs' - ones(2,1);
tpsf_coordinates = round( ones(N_coordinates, 2) + linspace(0.05, 0.95, N_coordinates)' * direction' );
cs_2d_mlfma_options.tpsf_indices = [tpsf_coordinates(:,1) - 1, tpsf_coordinates(:,2)] * [N_lattice_axis_cs(2); 1];
N_tpsf = numel(cs_2d_mlfma_options.tpsf_indices);

% create info string
str_tpsf_params = sprintf('%s_%d_%d_%.2f_%.2f_exc_%s_el%s_f_lb_%.2f_f_ub_%.2f_trans%s_abs_%s_thresh_%.2f', str_name, N_lattice_axis_cs(1), N_lattice_axis_cs(2), lattice_delta_x_cs * 1e4, lattice_delta_z_cs * 1e4, cs_2d_mlfma_options.excitation, str_elements_cs, f_lb_cs / 1e6, f_ub_cs / 1e6, str_transform, str_absorption, cs_2d_mlfma_options.normalize_columns_threshold * 1e2);

% compute TPSFs
[theta_tpsf, gamma_tpsf, column_norms, adjointness] = forward_simulation_mlfma_gpu_v18(zeros(N_lattice_axis_cs(2),N_lattice_axis_cs(1)), zeros(N_lattice_axis_cs(2),N_lattice_axis_cs(1)), A_in_td, f_lb_cs, f_ub_cs, N_elements, element_width, element_kerf, factor_interp_cs, lattice_delta_z_cs, lattice_pos_z_shift_cs, theta_incident_cw_cs, c_ref, f_s, cs_2d_mlfma_options);

% format and save results
if cs_2d_mlfma_options.material_parameter == 0
    
    % both material parameters
    
    % allocate memory
    theta_kappa_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    theta_rho_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    gamma_kappa_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    gamma_rho_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    
    % separate and plot TPSFs
    for index_tpsf = 1:N_tpsf
        
        % both material parameters  
        theta_kappa_tpsf(:, :, index_tpsf) = theta_tpsf(:, 1:N_lattice_axis_cs(1), index_tpsf);
        theta_rho_tpsf(:, :, index_tpsf) = theta_tpsf(:, (N_lattice_axis_cs(1) + 1):end, index_tpsf);
        
        gamma_kappa_tpsf(:, :, index_tpsf) = gamma_tpsf(:, 1:N_lattice_axis_cs(1), index_tpsf);
        gamma_rho_tpsf(:, :, index_tpsf) = gamma_tpsf(:, (N_lattice_axis_cs(1) + 1):end, index_tpsf);
        
        % logarithmic compression
        theta_kappa_tpsf_dB = 20*log10(abs(theta_kappa_tpsf(:, :, index_tpsf)) / max(max(abs(theta_kappa_tpsf(:, :, index_tpsf)))));
        theta_rho_tpsf_dB = 20*log10(abs(theta_rho_tpsf(:, :, index_tpsf)) / max(max(abs(theta_rho_tpsf(:, :, index_tpsf)))));
        
        gamma_kappa_tpsf_dB = 20*log10(abs(gamma_kappa_tpsf(:, :, index_tpsf)) / max(max(abs(gamma_kappa_tpsf(:, :, index_tpsf)))));
        gamma_rho_tpsf_dB = 20*log10(abs(gamma_rho_tpsf(:, :, index_tpsf)) / max(max(abs(gamma_rho_tpsf(:, :, index_tpsf)))));
    
        % difference tpsf
        gamma_tpsf_diff = gamma_kappa_tpsf(:, :, index_tpsf) - gamma_rho_tpsf(:, :, index_tpsf);
        gamma_tpsf_diff_dB = 20*log10(abs(gamma_tpsf_diff) / max(abs(gamma_tpsf_diff(:))));

        % indices of tpsf
        index_tpsf_x = ceil(cs_2d_mlfma_options.tpsf_indices(index_tpsf) / N_lattice_axis_cs(2));
        index_tpsf_z = cs_2d_mlfma_options.tpsf_indices(index_tpsf) - (index_tpsf_x - 1) * N_lattice_axis_cs(2);

        % display results
        figure(index_tpsf);
        % results for kappa
        subplot(3,3,1);
        imagesc(theta_kappa_tpsf_dB, [-dynamic_range_dB, 0]);
        if index_tpsf_x <= N_lattice_axis_cs(1);
            line([index_tpsf_x - 2, index_tpsf_x + 2], index_tpsf_z * ones(1,2), 'Color', 'r');
            line(index_tpsf_x * ones(1,2), [index_tpsf_z - 2, index_tpsf_z + 2], 'Color', 'r');
        end
        subplot(3,3,2);
        imagesc(lattice_pos_x_cs * 1e3, lattice_pos_z_cs * 1e3, gamma_kappa_tpsf_dB, [-dynamic_range_dB, 0]);
        subplot(3,3,3);
        gamma_kappa_tpsf_dft = fft2(gamma_kappa_tpsf(:, :, index_tpsf));
        gamma_kappa_tpsf_dft_dB = 20*log10(abs(gamma_kappa_tpsf_dft) / max(abs(gamma_kappa_tpsf_dft(:))));
        imagesc(axis_k_hat_x_cs, axis_k_hat_z_cs, fftshift(gamma_kappa_tpsf_dft_dB, 2), [-dynamic_range_dB, 0]);
        draw_FDT_cw(2*pi*f_lb_cs / c_ref, 2*pi*f_ub_cs / c_ref);
        % results for rho
        subplot(3,3,4);
        imagesc(theta_rho_tpsf_dB, [-dynamic_range_dB, 0]);
        if index_tpsf_x > N_lattice_axis_cs(1);
            line([index_tpsf_x - 2, index_tpsf_x + 2] - N_lattice_axis_cs(1), index_tpsf_z * ones(1,2), 'Color', 'r');
            line(index_tpsf_x * ones(1,2) - N_lattice_axis_cs(1), [index_tpsf_z - 2, index_tpsf_z + 2], 'Color', 'r');
        end
        subplot(3,3,5);
        imagesc(lattice_pos_x_cs * 1e3, lattice_pos_z_cs * 1e3, gamma_rho_tpsf_dB, [-dynamic_range_dB, 0]);
        subplot(3,3,6);
        gamma_rho_tpsf_dft = fft2(gamma_rho_tpsf(:, :, index_tpsf));
        gamma_rho_tpsf_dft_dB = 20*log10(abs(gamma_rho_tpsf_dft) / max(abs(gamma_rho_tpsf_dft(:))));
        imagesc(axis_k_hat_x_cs, axis_k_hat_z_cs, fftshift(gamma_rho_tpsf_dft_dB, 2), [-dynamic_range_dB, 0]);
        draw_FDT_cw(2*pi*f_lb_cs / c_ref, 2*pi*f_ub_cs / c_ref);
        subplot(3,3,7);
        imagesc(gamma_tpsf_diff_dB, [-dynamic_range_dB, 0]);
        colormap gray;
    end
        
    % save data as mat file
    str_filename = sprintf('%s/2d_tpsf_kappa_rho_%s.mat', str_path, str_tpsf_params);
    save(str_filename, 'theta_kappa_tpsf', 'gamma_kappa_tpsf', 'theta_rho_tpsf', 'gamma_rho_tpsf', 'N_tpsf', 'lattice_pos_x_cs', 'lattice_pos_z_cs', 'N_lattice_axis_cs', 'lattice_delta_x_cs', 'lattice_delta_z_cs', 'axis_k_hat_x_cs', 'axis_k_hat_z_cs', 'f_lb_cs', 'f_ub_cs', 'c_ref', 'indices_elements_cs', 'cs_2d_mlfma_options', 'column_norms', 'adjointness');

elseif cs_2d_mlfma_options.material_parameter == 1
        
    % only gamma_kappa
    
    % allocate memory
    theta_kappa_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    gamma_kappa_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
        
    % plot TPSFs
    for index_tpsf = 1:N_tpsf

        % only gamma_kappa
        theta_kappa_tpsf(:, :, index_tpsf) = theta_tpsf(:, 1:N_lattice_axis_cs(1), index_tpsf); 
        gamma_kappa_tpsf(:, :, index_tpsf) = gamma_tpsf(:, 1:N_lattice_axis_cs(1), index_tpsf);

        % logarithmic compression
        theta_kappa_tpsf_dB = 20*log10(abs(theta_kappa_tpsf(:, :, index_tpsf)) / max(max(abs(theta_kappa_tpsf(:, :, index_tpsf)))));
        gamma_kappa_tpsf_dB = 20*log10(abs(gamma_kappa_tpsf(:, :, index_tpsf)) / max(max(abs(gamma_kappa_tpsf(:, :, index_tpsf)))));

        % indices of tpsf
        index_tpsf_x = ceil(cs_2d_mlfma_options.tpsf_indices(index_tpsf) / N_lattice_axis_cs(2));
        index_tpsf_z = cs_2d_mlfma_options.tpsf_indices(index_tpsf) - (index_tpsf_x - 1) * N_lattice_axis_cs(2);

        % display results
        figure(index_tpsf);
        subplot(1,3,1);
        imagesc(theta_kappa_tpsf_dB, [-dynamic_range_dB, 0]);
        line([index_tpsf_x - 2, index_tpsf_x + 2], index_tpsf_z * ones(1,2), 'Color', 'r');
        line(index_tpsf_x * ones(1,2), [index_tpsf_z - 2, index_tpsf_z + 2], 'Color', 'r');
        subplot(1,3,2);
        imagesc(lattice_pos_x_cs * 1e3, lattice_pos_z_cs * 1e3, gamma_kappa_tpsf_dB, [-dynamic_range_dB, 0]);
        subplot(1,3,3);
        gamma_kappa_tpsf_dft = fft2(gamma_kappa_tpsf(:, :, index_tpsf));
        gamma_kappa_tpsf_dft_dB = 20*log10(abs(gamma_kappa_tpsf_dft) / max(abs(gamma_kappa_tpsf_dft(:))));
        imagesc(axis_k_hat_x_cs, axis_k_hat_z_cs, fftshift(gamma_kappa_tpsf_dft_dB, 2), [-dynamic_range_dB, 0]);
        draw_FDT_cw(2*pi*f_lb_cs / c_ref, 2*pi*f_ub_cs / c_ref);
        colormap gray;
    end
        
    % save data as mat file   
    str_filename = sprintf('%s/2d_tpsf_kappa_%s.mat', str_path, str_tpsf_params);
    save(str_filename, 'theta_kappa_tpsf', 'gamma_kappa_tpsf', 'N_tpsf', 'lattice_pos_x_cs', 'lattice_pos_z_cs', 'N_lattice_axis_cs', 'lattice_delta_x_cs', 'lattice_delta_z_cs', 'axis_k_hat_x_cs', 'axis_k_hat_z_cs', 'f_lb_cs', 'f_ub_cs', 'c_ref', 'indices_elements_cs', 'cs_2d_mlfma_options', 'column_norms', 'adjointness');

elseif cs_2d_mlfma_options.material_parameter == 2
        
	% only gamma_rho
	
    % allocate memory
    theta_rho_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    gamma_rho_tpsf = zeros(N_lattice_axis_cs(2), N_lattice_axis_cs(1), N_tpsf);
    
    % plot TPSFs
    for index_tpsf = 1:N_tpsf
        
        % only gamma_rho
        theta_rho_tpsf(:, :, index_tpsf) = theta_tpsf(:, (N_lattice_axis_cs(1) + 1):end, index_tpsf);
        gamma_rho_tpsf(:, :, index_tpsf) = gamma_tpsf(:, (N_lattice_axis_cs(1) + 1):end, index_tpsf);
        
        % logarithmic compression
        theta_rho_tpsf_dB = 20*log10(abs(theta_rho_tpsf(:, :, index_tpsf)) / max(max(abs(theta_rho_tpsf(:, :, index_tpsf)))));
        gamma_rho_tpsf_dB = 20*log10(abs(gamma_rho_tpsf(:, :, index_tpsf)) / max(max(abs(gamma_rho_tpsf(:, :, index_tpsf)))));

        % indices of tpsf
        index_tpsf_x = ceil(cs_2d_mlfma_options.tpsf_indices(index_tpsf) / N_lattice_axis_cs(2));
        index_tpsf_z = cs_2d_mlfma_options.tpsf_indices(index_tpsf) - (index_tpsf_x - 1) * N_lattice_axis_cs(2);

        % display results
        figure(index_tpsf);
        subplot(1,3,1);
        imagesc(theta_rho_tpsf_dB, [-dynamic_range_dB, 0]);
        line([index_tpsf_x - 2, index_tpsf_x + 2], index_tpsf_z * ones(1,2), 'Color', 'r');
        line(index_tpsf_x * ones(1,2), [index_tpsf_z - 2, index_tpsf_z + 2], 'Color', 'r');
        subplot(1,3,2);
        imagesc(lattice_pos_x_cs * 1e3, lattice_pos_z_cs * 1e3, gamma_rho_tpsf_dB, [-dynamic_range_dB, 0]);
        subplot(1,3,3);
        gamma_rho_tpsf_dft = fft2(gamma_rho_tpsf(:, :, index_tpsf));
        gamma_rho_tpsf_dft_dB = 20*log10(abs(gamma_rho_tpsf_dft) / max(abs(gamma_rho_tpsf_dft(:))));
        imagesc(axis_k_hat_x_cs, axis_k_hat_z_cs, fftshift(gamma_rho_tpsf_dft_dB, 2), [-dynamic_range_dB, 0]);
        draw_FDT_cw(2*pi*f_lb_cs / c_ref, 2*pi*f_ub_cs / c_ref);
        colormap gray;
    end
        
	% save data as mat file   
	str_filename = sprintf('%s/2d_tpsf_rho_%s.mat', str_path, str_tpsf_params);
	save(str_filename, 'theta_rho_tpsf', 'gamma_rho_tpsf', 'N_tpsf', 'lattice_pos_x_cs', 'lattice_pos_z_cs', 'N_lattice_axis_cs', 'lattice_delta_x_cs', 'lattice_delta_z_cs', 'axis_k_hat_x_cs', 'axis_k_hat_z_cs', 'f_lb_cs', 'f_ub_cs', 'c_ref', 'indices_elements_cs', 'cs_2d_mlfma_options', 'column_norms', 'adjointness');
end