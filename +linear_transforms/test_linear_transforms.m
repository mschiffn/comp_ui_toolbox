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