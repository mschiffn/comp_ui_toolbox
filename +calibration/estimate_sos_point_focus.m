function [ states, rel_RMSE ] = estimate_sos_point_focus( u_rx_tilde_qsw, xdc_array, states, options )
%
% Estimates the average speed of sound and the positions of
% multiple speckle regions or "beacons" (e.g. point-like reflectors).
%
% The function focuses
% the RF voltage signals on
% the initial location and analyzes
% the times-of-flight of
% the induced echoes using
% the inter-element cross-correlations.
% It iteratively corrects
% all phase-aberrations (see [1]) to enable
% sound speed estimation by
% the MATLAB Optimization Toolbox.
%
% INPUT:
%   u_rx_tilde_qsw = RF voltage signals obtained by complete SA sequence
%   xdc_array = transducer array
%   states = initial values of the average speed of sound and position of the ROI
%   options = calibration.options.sos_focus
%
% OUTPUT:
%   states = corrected input states
%   rel_RMSE = relative root mean-squared fitting error
%
% REFERENCES:
%	[1] S. W. Flax and M. O'Donnell, "Phase-Aberration Correction Using Signals From Point Reflectors and Diffuse Scatterers: Basic Principles",
%       IEEE TUFFC, Vol. 35, No. 6, Nov. 1988, pp. 758-767
%       DOI: 
%
% REMARKS:
%	- Ref. [1] claims that the iterative procedure works for speckle regions if the window length is long enough.
%
% author: Martin F. Schiffner
% date: 2020-01-21
% modified: 2020-01-22

    %----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure cell array for u_rx_tilde_qsw
	if ~iscell( u_rx_tilde_qsw )
        u_rx_tilde_qsw = { u_rx_tilde_qsw };
    end

	% ensure class scattering.sequences.setups.transducers.array_planar_regular
	if ~isa( xdc_array, 'scattering.sequences.setups.transducers.array_planar_regular' )
        errorStruct.message = 'xdc_array must be scattering.sequences.setups.transducers.array_planar_regular!';
        errorStruct.identifier = 'estimate_sos_point_focus:NoRegularPlanarArray';
        error( errorStruct );
    end

	% ensure cell array for states
	if ~iscell( states )
        states = { states };
    end

	% ensure cell array for options
    if ~iscell( options )
        options = { options };
    end

	% multiple u_rx_tilde_qsw / single xdc_array
	if ~isscalar( u_rx_tilde_qsw ) && isscalar( xdc_array )
        xdc_array = repmat( xdc_array, size( u_rx_tilde_qsw ) );
    end

	% multiple u_rx_tilde_qsw / single states
	if ~isscalar( u_rx_tilde_qsw ) && isscalar( states )
        states = repmat( states, size( u_rx_tilde_qsw ) );
    end

	% multiple u_rx_tilde_qsw / single options
	if ~isscalar( u_rx_tilde_qsw ) && isscalar( options )
        options = repmat( options, size( u_rx_tilde_qsw ) );
    end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( u_rx_tilde_qsw, xdc_array, states, options );

	%----------------------------------------------------------------------
	% 2.) process signal matrices
	%----------------------------------------------------------------------
	% specify cell arrays
	rel_RMSE = cell( size( u_rx_tilde_qsw ) );

	% iterate signal matrices
	for index_data = 1:numel( u_rx_tilde_qsw )

        %------------------------------------------------------------------
        % a) check arguments
        %------------------------------------------------------------------
        % ensure class processing.signal_matrix
        if ~isa( u_rx_tilde_qsw{ index_data }, 'processing.signal_matrix' )
            errorStruct.message = 'u_rx_tilde_qsw must be processing.signal_matrix!';
            errorStruct.identifier = 'estimate_sos_point_focus:NoSignalMatrices';
            error( errorStruct );
        end

        % ensure valid number of signal matrices
        if numel( u_rx_tilde_qsw{ index_data } ) ~= xdc_array( index_data ).N_elements
            errorStruct.message = sprintf( 'The number of elements in u_rx_tilde_qsw{ %d } must equal the number of elements in xdc_array( %d )!', index_data, index_data );
            errorStruct.identifier = 'estimate_sos_point_focus:InvalidNumberOfSignalMatrices';
            error( errorStruct );
        end

        % ensure valid numbers of signals
        if any( [ u_rx_tilde_qsw{ index_data }.N_signals ] ~= xdc_array( index_data ).N_elements )
            errorStruct.message = sprintf( 'The number of signals in u_rx_tilde_qsw( %d ) must equal the number of elements in xdc_array( %d )!', index_data, index_data );
            errorStruct.identifier = 'estimate_sos_point_focus:InvalidNumberOfSignals';
            error( errorStruct );
        end

        % method fourier_coefficients ensures equal subclasses of math.sequence_increasing_regular_quantized for u_rx_tilde_qsw{ index_data }.axis

        % ensure class calibration.state
        if ~isa( states{ index_data }, 'calibration.state' )
            errorStruct.message = sprintf( 'states{ %d } must be calibration.state!', index_data );
            errorStruct.identifier = 'estimate_sos_point_focus:NoStates';
            error( errorStruct );
        end

        % ensure class calibration.options.sos_focus
        if ~isa( options{ index_data }, 'calibration.options.sos_focus' )
            errorStruct.message = sprintf( 'options{ %d } must be calibration.options.sos_focus!', index_data );
            errorStruct.identifier = 'estimate_sos_point_focus:NoOptionsSoSFocus';
            error( errorStruct );
        end

        % multiple states{ index_data } / single options{ index_data }
        if ~isscalar( states{ index_data } ) && isscalar( options{ index_data } )
            options{ index_data } = repmat( options{ index_data }, size( states{ index_data } ) );
        end

        % ensure equal number of dimensions and sizes
        auxiliary.mustBeEqualSize( states{ index_data }, options{ index_data } );

        %------------------------------------------------------------------
        % b) predict TOFs for all states
        %------------------------------------------------------------------
        [ times_of_flight_init, times_of_flight_sc_init ] = calibration.function_tof_qsw( xdc_array( index_data ).positions_ctr, states{ index_data } );

        % ensure cell array for times_of_flight_init
        if ~iscell( times_of_flight_init )
            times_of_flight_init = { times_of_flight_init };
            times_of_flight_sc_init = { times_of_flight_sc_init };
        end

        %------------------------------------------------------------------
        % c) estimate speed of sound for each point-like target
        %------------------------------------------------------------------
        % number of point-like targets
        N_targets = numel( states{ index_data } );

        % initialize results of nonlinear LSE w/ zeros
        pos_r0_est = physical_values.meter( zeros( N_targets, 3 ) );
        c_avg_est = physical_values.meter_per_second( zeros( N_targets, 1 ) );

        % initialize relative RMSE of linear LSE
        rel_RMSE{ index_data } = zeros( size( states{ index_data } ) );

        % iterate point-like targets
        for index_target = 1:N_targets

            % display status
            fprintf( 'index_target = %d of %d...\n\n', index_target, N_targets );

            %--------------------------------------------------------------
            % i.) initialize refocusing sequence
            %--------------------------------------------------------------
% TODO: use F-number
            % indices of active elements
            indices_elements_active = options{ index_data }( index_target ).indices_elements_tx;

            % number of active elements
            N_elements_active = numel( indices_elements_active );

            % initial focusing delays
            delays_act = times_of_flight_sc_init{ index_target }( indices_elements_active );
            [ val_min, index_min ] = min( delays_act );
            delays_act = delays_act - val_min;

            %--------------------------------------------------------------
            % ii.) compute time intervals based on predicted TOFs and waveform center
            %--------------------------------------------------------------
            % predict waveform centers using TOFs
            times_of_flight_init_ctr = times_of_flight_init{ index_target }( indices_elements_active( index_min ), indices_elements_active ) + options{ index_data }( index_target ).time_shift_ctr;

            % create time intervals
            intervals_t = move( options{ index_data }( index_target ).interval_window_t, times_of_flight_init_ctr );

            %--------------------------------------------------------------
            % iii.) compute Fourier coefficients for refocusing
            %--------------------------------------------------------------
            % extract signal lengths
            Ts_rec = cellfun( @( x ) x.members( end ) - x.members( 1 ), { u_rx_tilde_qsw{ index_data }( indices_elements_active ).axis } );
            deltas_t = cellfun( @( x ) x.delta, { u_rx_tilde_qsw{ index_data }( indices_elements_active ).axis } );
            delta_t = unique( deltas_t );
% unique deltas?
% TODO: time axis in u_rx_tilde_qsw{ index_data } and delays_act for offset?

            % determine length for Fourier analysis
            T_analysis = ceil( ( max( Ts_rec ) + max( delays_act ) ) / delta_t ) * delta_t;

            % compute Fourier coefficients
% TODO: include only relevant rx elements!
            u_rx_qsw = fourier_coefficients( u_rx_tilde_qsw{ index_data }( indices_elements_active ), T_analysis, options{ index_data }( index_target ).interval_f );

            %--------------------------------------------------------------
            % iv.) refocusing sequence
            %--------------------------------------------------------------
            % start refocusing sequence
            for index_iteration = 1:options{ index_data }( index_target ).N_iterations_max

                % display status
                fprintf( '\tindex_iteration = %d of %d...', index_iteration, options{ index_data }( index_target ).N_iterations_max );

                %----------------------------------------------------------
                % a) focus RF voltage signals based on delays
                %----------------------------------------------------------
                % compute phase shifts
                phase_shift = processing.signal( u_rx_qsw( 1 ).axis, mat2cell( exp( 2j * pi * u_rx_qsw( 1 ).axis.members * delays_act' ), abs( u_rx_qsw( 1 ).axis ), ones( 1, N_elements_active ) ) );

                % apply phase shifts and interpolation
                u_rx_tilde_focus_int = signal( sum( u_rx_qsw .* phase_shift' ), double( u_rx_tilde_qsw{ index_data }( 1 ).axis.q_lb ) * options{ index_data }( index_target ).factor_interp, u_rx_tilde_qsw{ index_data }( 1 ).axis.delta / options{ index_data }( index_target ).factor_interp );

                %----------------------------------------------------------
                % b) interpolate focused RF voltage signals and cut out waveforms
                %----------------------------------------------------------
                % cut out waveforms (apply windows)
                u_rx_tilde_focus_int_window = cut_out( u_rx_tilde_focus_int, cat( 1, intervals_t.lb ), cat( 1, intervals_t.ub ), num2cell( indices_elements_active ), options{ index_data }( index_target ).setting_window );

                % illustrate cut out
                figure( 998 );
                imagesc( indices_elements_active, double( u_rx_tilde_focus_int.axis.members ), illustration.dB( hilbert( u_rx_tilde_focus_int.samples( :, indices_elements_active ) ), 20 ), [ -60, 0 ] );
                line( indices_elements_active, double( [ intervals_t.lb ] ), 'Color', [ 1, 1, 0.99 ], 'LineWidth', 2, 'LineStyle', ':' );
                line( indices_elements_active, double( [ intervals_t.ub ] ), 'Color', [ 1, 1, 0.99 ], 'LineWidth', 2, 'LineStyle', ':' );

                %----------------------------------------------------------
                % c) compute inter-element lags
                %----------------------------------------------------------
                % initialize lags with zeros
                lags_adjacent = physical_values.second( zeros( N_elements_active, 1 ) );
                corr_vals = zeros( N_elements_active, 1 );

                % iterate specified rx elements
                for index_selected_rx = 2:N_elements_active

                    % extract RF data of adjacent channels
                    u_rx_tilde_qsw_int_window_act = u_rx_tilde_focus_int_window( index_selected_rx );
                    u_rx_tilde_qsw_int_window_prev = u_rx_tilde_focus_int_window( index_selected_rx - 1 );

                    % compute inter-element correlation coefficients
                    [ data_pw_int_cut_corr, data_pw_int_cut_corr_lags ] = xcorr( u_rx_tilde_qsw_int_window_act.samples / norm( u_rx_tilde_qsw_int_window_act.samples ), u_rx_tilde_qsw_int_window_prev.samples / norm( u_rx_tilde_qsw_int_window_prev.samples ) );

                    % detect and save maximum of cross-correlation
                    [ corr_vals( index_selected_rx ), index_max ] = max( data_pw_int_cut_corr );

                    % estimate relative time delays
                    lags_adjacent( index_selected_rx ) = data_pw_int_cut_corr_lags( index_max ) * u_rx_tilde_qsw_int_window_act.axis.delta + u_rx_tilde_qsw_int_window_act.axis.members( 1 ) - u_rx_tilde_qsw_int_window_prev.axis.members( 1 );

                    % illustrate inter-element lag
%                     figure( 999 );
%                     plot( u_rx_tilde_qsw_int_window_act.axis.members - lags_adjacent( index_selected_rx ), u_rx_tilde_qsw_int_window_act.samples / max( u_rx_tilde_qsw_int_window_act.samples ), u_rx_tilde_qsw_int_window_prev.axis.members, u_rx_tilde_qsw_int_window_prev.samples / max( u_rx_tilde_qsw_int_window_prev.samples ) );
%                     pause( 0.1 );

                end % for index_selected_rx = 2:N_elements_active

                %----------------------------------------------------------
                % d) estimate TOFs
                %----------------------------------------------------------
                % integrate inter-element delays and find lateral position of minimum
                lags_adjacent_cs = cumsum( lags_adjacent, 1 );
                [ lags_adjacent_cs_min, index_min ] = min( lags_adjacent_cs, [], 1 );
                lags_adjacent_cs = lags_adjacent_cs - lags_adjacent_cs_min;

                % find maximum of envelope of interpolated RF data
                u_rx_tilde_focus_int_window_env = abs( hilbert( u_rx_tilde_focus_int_window( index_min ).samples ) );
                [ ~, index_max ] = max( u_rx_tilde_focus_int_window_env );
                time_max = u_rx_tilde_focus_int_window( index_min ).axis.members( index_max );

                % estimate minimum time-of-flight
                tof_min = time_max - options{ index_data }( index_target ).time_shift_ctr;

                % estimate TOFs for all rx elements
                tofs_act = tof_min + lags_adjacent_cs;

                %----------------------------------------------------------
                % e) update computed delays with estimated TOFs:
                %----------------------------------------------------------
                % compute residual and relative RMSE
                delays_error = lags_adjacent_cs - delays_act;
                delays_rel_RMSE = norm( delays_error ) / norm( delays_act );

                % update delays
                delays_act = lags_adjacent_cs;

                % display status
                fprintf( 'done! (rel. RMSE = %.2f %%)\n', delays_rel_RMSE * 1e2 );

                %----------------------------------------------------------
                % f) termination condition
                %----------------------------------------------------------
                if delays_rel_RMSE <= options{ index_data }( index_target ).rel_RMSE
                    break;
                end

            end % for index_iteration = 1:options{ index_data }( index_target ).N_iterations_max

            %--------------------------------------------------------------
            % v.) nonlinear estimate
            %--------------------------------------------------------------
            % initial state
            theta_0 = [ double( states{ index_data }( index_target ).position_target ), double( states{ index_data }( index_target ).c_avg ) ];

            % boundaries
            theta_lbs = [ -2e-2, -5e-3, 0,    50 ];
            theta_ubs = [  2e-2,  5e-3, 8e-2, 2000 ];

            % set optimization options
            options_optimization = optimoptions( 'lsqcurvefit', 'Algorithm', 'trust-region-reflective', 'FunValCheck', 'on', 'Diagnostics', 'off', 'Display', 'off', 'FunctionTolerance', 1e-10, 'OptimalityTolerance', 1e-10, 'StepTolerance', 1e-10, 'SpecifyObjectiveGradient', true, 'CheckGradients', false, 'FiniteDifferenceType', 'central', 'FiniteDifferenceStepSize', 1e-10, 'MaxFunctionEvaluations', 5e3, 'MaxIterations', 5e3 );

            % find solutions to nonlinear least squares problems
            [ theta_tof, resnorm, residual, exitflag, output ] = lsqcurvefit( @tof_us, theta_0, double( xdc_array( index_data ).positions_ctr( indices_elements_active, : ) ), double( tofs_act ) * 1e6, theta_lbs, theta_ubs, options_optimization );

            % extract target position and speed of sound
            pos_r0_est( index_target, : ) = physical_values.meter( theta_tof( 1:3 ) );
            c_avg_est( index_target ) = physical_values.meter_per_second( theta_tof( 4 ) );

            % compute estimation error
            tofs_act_error_init_us = double( tofs_act ) * 1e6 - tof_us( theta_0, double( xdc_array( index_data ).positions_ctr( indices_elements_active, : ) ) );
            tofs_act_error_us = double( tofs_act ) * 1e6 - tof_us( theta_tof, double( xdc_array( index_data ).positions_ctr( indices_elements_active, : ) ) );

            rel_RMSE_0 = norm( tofs_act_error_init_us ) ./ norm( double( tofs_act ) * 1e6 );
            rel_RMSE{ index_data }( index_target ) = norm( tofs_act_error_us ) ./ norm( double( tofs_act ) * 1e6 );

            % display status
            fprintf( '\ndone! (speed of sound = %.2f m/s)\n\n', c_avg_est( index_target ) );

            % check for improvement
            if rel_RMSE{ index_data }( index_target ) > rel_RMSE_0
                warning('Nonlinear estimate did not improve the results!');
            end

            %--------------------------------------------------------------
            % vi.) illustration
            %--------------------------------------------------------------
            figure( index_data );
            subplot( 3, N_targets, index_target );
            plot( indices_elements_active, double( tofs_act ) * 1e6 );
            title( 'Detected' );
            subplot( 3, N_targets, N_targets + index_target );
            plot( indices_elements_active, double( tof_us( theta_tof, double( xdc_array( index_data ).positions_ctr( indices_elements_active, : ) ) ) ) );
            title( sprintf( 'Estimated ( %.2f m/s)', double( c_avg_est( index_target ) ) ) );
            subplot( 3, N_targets, 2 * N_targets + index_target );
            plot( indices_elements_active, tofs_act_error_us );
            title( sprintf( 'Error (%.2f %%, N_iterations = %d)', rel_RMSE{ index_data }( index_target ) * 1e2, index_iteration ), 'Interpreter', 'none' );

        end % for index_target = 1:N_targets

        %------------------------------------------------------------------
        % d.) create estimated states
        %------------------------------------------------------------------
        % create estimated states
        states{ index_data } = calibration.state( pos_r0_est, c_avg_est );

	end % for index_data = 1:numel( u_rx_tilde_qsw )

	% avoid cell arrays for single u_rx_tilde_qsw
	if isscalar( u_rx_tilde_qsw )
        states = states{ 1 };
        rel_RMSE = rel_RMSE{ 1 };
    end

	%----------------------------------------------------------------------
	% compute TOFs (microseconds)
	%----------------------------------------------------------------------
	function [ y, J ] = tof_us( theta, positions )

        % compute distances
        vect_r0_r = [ positions, zeros( size( positions, 1 ), 1 ) ] - theta( 1:( end - 1 ) );
        dist = vecnorm( vect_r0_r, 2, 2 );

        % compute round-trip times-of-flight (us)
        y = 1e6 * ( theta( end - 1 ) + dist ) / theta( end );

        % check if Jacobian is required
        if nargout > 1

            % compute Jacobian
            J = zeros( numel( y ), numel( theta ) );

            % partial derivatives w/ respect to lateral positions
            for index_dim = 1:( numel( theta ) - 2 )
                temp = - vect_r0_r( :, index_dim ) ./ dist;
                J( :, index_dim ) = 1e6 * temp / theta( end );
            end

            % partial derivative w/ respect to axial position
            temp = - vect_r0_r( :, end ) ./ dist;
            J( :, end - 1 ) = 1e6 * ( 1 + temp ) / theta( end );

            % partial derivative w/ respect to SoS
            J( :, end ) = - y( : ) / theta( end );

        end % if nargout > 1

    end % function [ y, J ] = tof_us( theta, positions )

end % function [ states, rel_RMSE ] = estimate_sos_point_focus( u_rx_tilde_qsw, xdc_array, states, options )
