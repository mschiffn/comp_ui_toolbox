function p_in_grad = compute_p_in_gradient( scan_configuration, setting_tx, set_f )
%
% compute gradients of the incident acoustic pressure fields associated with
% a superposition of quasi-(d-1)-spherical waves
%
% assumption: fields are laterally symmetric about center axis of the physical element
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-01-16
%

    N_interp_rx
    FOV_N_points
    N_samples_f
    A_in_analy_cropped
    D_math_ref

    %----------------------------------------------------------------------
	% 1.) compute reference fields radiated by the first array element
	%----------------------------------------------------------------------
    p_in_grad_ref_x = zeros( N_interp_rx, FOV_N_points, N_samples_f );
	p_in_grad_ref_z = zeros( N_interp_rx, FOV_N_points, N_samples_f );
    e_r_minus_r_s_ref_x = ( repmat( FOV_pos( :, 1 )', [N_interp_rx, 1] ) - repmat( XDC_pos_x( 1:N_interp_rx )', [1, FOV_N_points] ) ) ./ D_math_ref;
	e_r_minus_r_s_ref_z = repmat( FOV_pos( :, 2 )', [N_interp_rx, 1] ) ./ D_math_ref;

	for index_f = 1:N_samples_f
        temp = -0.25j * axis_k_tilde( index_f ) * A_in_analy_cropped( index_f ) * besselh( 1, 2, axis_k_tilde( index_f ) * D_math_ref );
        p_in_grad_ref_x( :, :, index_f ) = temp .* e_r_minus_r_s_ref_x;
        p_in_grad_ref_z( :, :, index_f ) = temp .* e_r_minus_r_s_ref_z;
    end
	p_in_grad_ref_x = reshape( squeeze( sum( p_in_grad_ref_x, 1 ) ), [FOV_N_points_axis(2), FOV_N_points_axis(1), N_samples_f] );
	p_in_grad_ref_z = reshape( squeeze( sum( p_in_grad_ref_z, 1 ) ), [FOV_N_points_axis(2), FOV_N_points_axis(1), N_samples_f] );

    %----------------------------------------------------------------------
	% 2.) compute phase shifts
	%----------------------------------------------------------------------
    % quantize time delays
	time_delays_quantized = round( setting_tx.time_delays * scan_configuration.f_clk ) / scan_configuration.f_clk;
	shift_phase = exp( -1j * axis_omega(:) * time_delays_quantized );

	% initialize values of the gradient of the incident acoustic pressure with zeros
	for index_f = 1:N_samples_f
        p_in_grad{ index_f, 1 } = zeros( FOV_N_points_axis(2), FOV_N_points_axis(1) );
        p_in_grad{ index_f, 2 } = zeros( FOV_N_points_axis(2), FOV_N_points_axis(1) );
    end

    %----------------------------------------------------------------------
	% 3.) superimpose gradients of the quasi-(d-1)-spherical waves
    %----------------------------------------------------------------------
	for index_element = setting_tx.indices_active

        % shift in grid points required for current mathematical tx element
        delta_lattice_points = ( index_element - 1 ) * factor_interp_tx;

        % compute summand for the incident pressure field
        index_start = FOV_N_points_axis( 1 ) - ( XDC_N_elements - 1 ) * factor_interp_tx + 1;
        index_stop = index_start + delta_lattice_points - 1;
        p_incident_grad_summand_x = [ -p_in_grad_ref_x( :, index_stop:-1:index_start, : ), p_in_grad_ref_x( :, 1:(end - delta_lattice_points), : ) ];
        p_incident_grad_summand_z = [ p_in_grad_ref_z( :, index_stop:-1:index_start, : ), p_in_grad_ref_z( :, 1:(end - delta_lattice_points), : ) ];
%         rel_RMSE_x( index_element ) = norm( p_incident_grad_summand_x(:) - p_incident_grad_act_x(:) ) / norm( p_incident_grad_act_x(:) ) * 1e2;
%         rel_RMSE_z( index_element ) = norm( p_incident_grad_summand_z(:) - p_incident_grad_act_z(:) ) / norm( p_incident_grad_act_z(:) ) * 1e2;
        p_incident_grad_summand_x = p_incident_grad_summand_x .* repmat( reshape( shift_phase( :, index_element ), [1, 1, N_samples_f] ), [FOV_N_points_axis(2), FOV_N_points_axis(1), 1] );
        p_incident_grad_summand_x = p_incident_grad_summand_x * apodization_weights( index_element );
        p_incident_grad_summand_z = p_incident_grad_summand_z .* repmat( reshape( shift_phase( :, index_element ), [1, 1, N_samples_f] ), [FOV_N_points_axis(2), FOV_N_points_axis(1), 1] );
        p_incident_grad_summand_z = p_incident_grad_summand_z * apodization_weights( index_element );

        for index_f = 1:N_samples_f
            p_in_grad{ index_f, 1 } = p_in_grad{ index_f, 1 } + p_incident_grad_summand_x( :, :, index_f );
            p_in_grad{ index_f, 2 } = p_in_grad{ index_f, 2 } + p_incident_grad_summand_z( :, :, index_f );
        end

        figure( N_samples_f );
        subplot(1,2,1);
        imagesc( abs( p_in_grad{ index_f_low, 1 } ) );
        subplot(1,2,2);
        imagesc( abs( p_in_grad{ index_f_low, 2 } ) );
	end % for index_element = setting_tx.indices_active
    
    % analyze gradient of incident pressure
    print_gradient(p_incident_grad, index_f_low, index_f_high, index_incident);
end
	