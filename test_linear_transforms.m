% test classes for linear transform
% author: Martin Schiffner
% date: 2016-08-13
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_cases = 25;
N_points = 512;
N_lattice_axis = [N_points, N_points];
weights = ones( N_points^2, 1 ) + 1j * ones( N_points^2, 1 );

%--------------------------------------------------------------------------
% define transforms
%--------------------------------------------------------------------------
% orthonormal transforms
LT_identity                 = linear_transforms.identity( N_lattice_axis );
LT_fourier                  = linear_transforms.fourier( N_lattice_axis );
LT_fourier_block            = linear_transforms.fourier_block( N_lattice_axis, [32, 32] );
LT_wavelet_haar             = linear_transforms.wavelet( 'Haar', 0, N_points, 0 );
LT_wavelet_db4              = linear_transforms.wavelet( 'Daubechies', 4, N_points, 0 );
LT_wavelet_db10             = linear_transforms.wavelet( 'Daubechies', 10, N_points, 0 );
LT_wavelet_db20             = linear_transforms.wavelet( 'Daubechies', 20, N_points, 0 );

options.qmf = MakeONFilter( 'Daubechies', 4 );
options.scale_coarsest = 0;
op_psi = @(x, mode) psi_wavelet( N_lattice_axis, x, mode, options );

% invertible transforms
LT_weighting_1              = linear_transforms.weighting( weights );

% linear transforms
LT_wave_atom_ortho          = linear_transforms.wave_atom( N_lattice_axis, 'ortho' );
LT_wave_atom_directional	= linear_transforms.wave_atom( N_lattice_axis, 'directional' );
LT_wave_atom_complex        = linear_transforms.wave_atom( N_lattice_axis, 'complex' );
LT_curvelet                 = linear_transforms.curvelet( N_lattice_axis );

% concatenated transforms
LT_concatenate_vertical     = linear_transforms.concatenate_vertical( LT_identity, LT_weighting_1, LT_fourier, LT_wavelet_db10 );
LT_concatenate_diagonal     = linear_transforms.concatenate_diagonal( LT_identity, LT_weighting_1, LT_fourier, LT_wavelet_db10 );

% composite transforms
size_transform = LT_concatenate_vertical.size_transform;
N_weights_2 = size_transform(1);
weights_2 = ones( N_weights_2, 1 ) + 1j * ones( N_weights_2, 1 );
LT_weighting_2          = linear_transforms.weighting( weights_2 );
LT_composite            = linear_transforms.composition( LT_weighting_2, LT_concatenate_vertical, LT_weighting_1 );

%--------------------------------------------------------------------------
% setup test
%--------------------------------------------------------------------------
transforms = { LT_identity, LT_fourier, LT_fourier_block, LT_wavelet_haar, LT_wavelet_db4, LT_wavelet_db10, LT_wavelet_db20, LT_weighting_1, LT_wave_atom_ortho, LT_wave_atom_directional, LT_wave_atom_complex, LT_curvelet, LT_concatenate_vertical, LT_concatenate_diagonal, LT_composite };
N_transforms = numel( transforms );

error_fwd_inv	= zeros( N_transforms, N_cases );
error_inv_fwd	= zeros( N_transforms, N_cases );
error_adj       = zeros( N_transforms, N_cases );
norms_cols      = zeros( N_transforms, N_cases );

for index_transform = 1:N_transforms
    
	% print status information
    fprintf( 'index_transform = %d of %d (%s):', index_transform, N_transforms, transforms{ index_transform }.str_name );

    % get size of transform
    size_transform	= transforms{ index_transform }.operator_transform( [], 0 );
    N_coefficients	= size_transform(1);
    N_lattice       = size_transform(2);
    indices_rand	= randperm( N_coefficients );

    % iterate over random cases
    for index_case = 1:N_cases

        % print case index
        fprintf( ' %d', index_case );

        %------------------------------------------------------------------
        % forward and inverse transforms
        %------------------------------------------------------------------
        % generate random test vector
        test_lat = randn(N_lattice, 1) + 1j * randn(N_lattice, 1);
        test_lat_norm = norm( test_lat(:), 2 );

        % compute forward transform
        test_lat_fwd = transforms{ index_transform }.operator_transform( test_lat, 1 );
%         test_lat_fwd_old = op_psi( test_lat, 1 );

        if isa( transforms{ index_transform }, 'linear_transforms.invertible_linear_transform' )

            % compute inverse transform
            test_lat_inv = transforms{ index_transform }.inverse_transform( test_lat_fwd );
%             test_lat_inv_old = op_psi( test_lat_fwd, 2 );

            % compute rel. RMSEs
            error_fwd_inv( index_transform, index_case ) = norm( test_lat_inv(:) - test_lat(:) ) / test_lat_norm;
        end

        %------------------------------------------------------------------
        % adjoint and inverse transforms
        %------------------------------------------------------------------
        % generate random test vector
        test_coef = randn(N_coefficients , 1) + 1j * randn(N_coefficients, 1);
        test_coef_norm = norm( test_coef(:), 2 );

        % compute adjoint transform
        test_coef_adj = transforms{ index_transform }.operator_transform( test_coef, 2 );
        
        if isa( transforms{ index_transform }, 'linear_transforms.invertible_linear_transform' )

            % compute inverse transform
            test_coef_inv = transforms{ index_transform }.inverse_transform( test_coef );

            % compute forward transform
            test_coef_fwd = transforms{ index_transform }.operator_transform( test_coef_inv, 1 );

            % compute rel. RMSEs
            error_inv_fwd( index_transform, index_case ) = norm( test_coef_fwd(:) - test_coef(:) ) / test_coef_norm;
        end

        %------------------------------------------------------------------
        % adjoint test
        %------------------------------------------------------------------
        error_adj( index_transform, index_case ) = test_lat_fwd.' * conj( test_coef(:) ) - test_lat.' * conj( test_coef_adj(:) );

        %------------------------------------------------------------------
        % column norms
        %------------------------------------------------------------------
        test_coef = zeros( N_coefficients, 1 );
        test_coef( indices_rand(index_case) ) = 1;
        
        % compute adjoint transform
        test_coef_adj = transforms{ index_transform }.operator_transform( test_coef, 2 );
        
        norms_cols( index_transform, index_case ) = norm( test_coef_adj(:) );
    end
    
    fprintf( '\n');
end

% statistics of results
error_fwd_inv_mean	= mean( error_fwd_inv, 2 ) * 1e2;
error_adj_fwd_mean	= mean( error_inv_fwd, 2 ) * 1e2;
error_adj_mean      = mean( error_adj, 2 );
norms_cols_mean     = mean( norms_cols, 2 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% test convolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 0.) parameters
%--------------------------------------------------------------------------
N_signals = 128;
T_s = physical_values.second( 1 / 40e6 );
N_samples = 3200;
N_samples_shift = 78;
axis_t = math.sequence_increasing_regular( N_samples_shift, N_samples_shift + N_samples - 1, T_s );
f_0 = physical_values.hertz( 3e6 );

%--------------------------------------------------------------------------
% 1.) create bandpass signal
%--------------------------------------------------------------------------
samples_BP_tilde = sin( 2 * pi * f_0 * ( axis_t.members - N_samples_shift * T_s ) );
signal_BP_tilde = discretizations.signal_matrix( axis_t, repmat( samples_BP_tilde, [ 1, N_signals ] ) );

%--------------------------------------------------------------------------
% 2.) create time-dependent variable gain
%--------------------------------------------------------------------------
TGC_curve = regularization.tgc.exponential( math.interval_quantized( axis_t.q_lb, axis_t.q_ub + 1, axis_t.delta ), physical_values.hertz( 2e4 ) );

% sample TGC curve
samples_gain_tilde = sample_curve( TGC_curve, axis_t );
signal_gain_tilde = discretizations.signal( axis_t, samples_gain_tilde );

%--------------------------------------------------------------------------
% 3.) apply TGC in the time domain
%--------------------------------------------------------------------------
signal_BP_tgc_tilde = signal_BP_tilde .* signal_gain_tilde;

%--------------------------------------------------------------------------
% 4.) apply TGC in the frequency domain (convolution)
%--------------------------------------------------------------------------
% Fourier coefficients (numeric)
signal_BP = fourier_coefficients( signal_BP_tilde );
signal_gain = fourier_coefficients( signal_gain_tilde );

% Fourier coefficients (analytic)
TGC_curve_coef = fourier_coefficients( TGC_curve, 10 * abs( axis_t ) * T_s, -60 );
M_kernel = abs( TGC_curve_coef.axis ) - 1;

% error
rel_RMSE_coef = norm( signal_gain.samples - TGC_curve_coef.samples ) / norm( TGC_curve_coef.samples );

% define convolution
kernel = [ conj( TGC_curve_coef.samples( end:-1:2 ) ); TGC_curve_coef.samples ];
LT_convolution = linear_transforms.convolution( kernel, abs( signal_BP.axis ) );

% apply convolution
[ samples_BP_tgc_conv_dft, samples_BP_tgc_conv_mat ] = forward_transform( LT_convolution, signal_BP.samples );

% apply adjoint convolution
[ samples_BP_tgc_conv_adj_dft, samples_BP_tgc_conv_adj_mat ] = adjoint_transform( LT_convolution, samples_BP_tgc_conv_dft );

% errors
rel_RMSE_fwd = norm( samples_BP_tgc_conv_dft( : ) - samples_BP_tgc_conv_mat( : ) ) / norm( samples_BP_tgc_conv_mat( : ) );
rel_RMSE_adj = norm( samples_BP_tgc_conv_adj_dft( : ) - samples_BP_tgc_conv_adj_mat( : ) ) / norm( samples_BP_tgc_conv_adj_mat( : ) );

% create signal matrix
signal_BP_tgc_conv_dft = discretizations.signal_matrix( signal_BP.axis, samples_BP_tgc_conv_dft );

% time-domain signals
signal_BP_tgc_conv_dft_tilde = signal( signal_BP_tgc_conv_dft, N_samples_shift, T_s );

% relative RMSEs
rel_RMSE_dft = norm( signal_BP_tgc_conv_dft_tilde.samples - signal_BP_tgc_tilde.samples ) / norm( signal_BP_tgc_tilde.samples );

figure( 1 );
plot( double( signal_BP_tgc_conv_dft_tilde.axis.members ), signal_BP_tgc_conv_dft_tilde.samples, signal_BP_tgc_tilde.axis.members, signal_BP_tgc_tilde.samples );
title( sprintf( 'rel. RMSE = %.2f %%', rel_RMSE_dft * 1e2 ) );

%--------------------------------------------------------------------------
% b) test for adjointness
%--------------------------------------------------------------------------
% specify test vectors
test_in = randn( LT_convolution.N_points, 1 ) + 1j * randn( LT_convolution.N_points, 1 );
test_out = randn( LT_convolution.N_coefficients, 1 ) + 1j * randn( LT_convolution.N_coefficients, 1 );

% adjointness must be close to zero
adjointness = ( test_out' * forward_transform( LT_convolution, test_in ) - adjoint_transform( LT_convolution, test_out )' * test_in ) / ( test_out' * forward_transform( LT_convolution, test_in ) );
