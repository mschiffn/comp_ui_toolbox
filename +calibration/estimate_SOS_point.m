function [ states, rel_RMSE, pulse_shape_mean, pulse_shape_std_dev ] = estimate_SOS_point( u_rx_tilde_qpw, xdc_array, states, options )
%
% estimate the average speed of sound using
% the inter-element cross-correlations for
% multiple point-like targets (cf. [1], [2])
%
% [1] M. E. Anderson and G. E. Trahey, "The direct estimation of sound speed using pulse–echo ultrasound",
%	  J. Acoust. Soc. Am., Nov. 1998, Vol. 104, No. 5, pp. 3099-3106
% [2] S. W. Flax and M. O'Donnell, "Phase-Aberration Correction Using Signals From Point Reflectors and Diffuse Scatterers: Basic Principles",
%     IEEE TUFFC, Vol. 35, No. 6, Nov. 1988, pp. 758-767
%
% author: Martin F. Schiffner
% date: 2014-09-20
% modified: 2019-11-14

    %----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure class discretizations.signal_matrix
	if ~isa( u_rx_tilde_qpw, 'discretizations.signal_matrix' )
        errorStruct.message = 'u_rx_tilde_qpw must be discretizations.signal_matrix!';
        errorStruct.identifier = 'estimate_SOS_point:NoSignalMatrices';
        error( errorStruct );
    end

    % ensure class scattering.sequences.setups.transducers.array_planar_regular
	if ~isa( xdc_array, 'scattering.sequences.setups.transducers.array_planar_regular' )
        errorStruct.message = 'xdc_array must be scattering.sequences.setups.transducers.array_planar_regular!';
        errorStruct.identifier = 'estimate_SOS_point:NoRegularPlanarArray';
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

	% multiple u_rx_tilde_qpw / single xdc_array
	if ~isscalar( u_rx_tilde_qpw ) && isscalar( xdc_array )
        xdc_array = repmat( xdc_array, size( u_rx_tilde_qpw ) );
    end

    % multiple u_rx_tilde_qpw / single states
	if ~isscalar( u_rx_tilde_qpw ) && isscalar( states )
        states = repmat( states, size( u_rx_tilde_qpw ) );
    end

	% multiple u_rx_tilde_qpw / single options
	if ~isscalar( u_rx_tilde_qpw ) && isscalar( options )
        options = repmat( options, size( u_rx_tilde_qpw ) );
    end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( u_rx_tilde_qpw, xdc_array, states, options );

    %----------------------------------------------------------------------
	% 2.) process signal matrices
	%----------------------------------------------------------------------
	% specify cell arrays
	rel_RMSE = cell( size( u_rx_tilde_qpw ) );
    pulse_shape = cell( size( u_rx_tilde_qpw ) );
	pulse_shape_mean = cell( size( u_rx_tilde_qpw ) );
	pulse_shape_std_dev = cell( size( u_rx_tilde_qpw ) );

    % iterate signal matrices
    for index_data = 1:numel( u_rx_tilde_qpw )

        %------------------------------------------------------------------
        % a) check arguments
        %------------------------------------------------------------------
        % ensure valid number of signals
        if u_rx_tilde_qpw( index_data ).N_signals ~= xdc_array( index_data ).N_elements
            errorStruct.message = sprintf( 'The number of signals in u_rx_tilde_qpw( %d ) must equal the number of elements in xdc_array( %d )!', index_data, index_data );
            errorStruct.identifier = 'estimate_SOS_point:InvalidNumberOfSignals';
            error( errorStruct );
        end

        % ensure class calibration.state
        if ~isa( states{ index_data }, 'calibration.state' )
            errorStruct.message = sprintf( 'states{ %d } must be calibration.state!', index_data );
            errorStruct.identifier = 'estimate_SOS_point:NoStates';
            error( errorStruct );
        end

        % ensure class calibration.options
        if ~isa( options{ index_data }, 'calibration.options' )
            errorStruct.message = sprintf( 'options{ %d } must be calibration.options!', index_data );
            errorStruct.identifier = 'estimate_SOS_point:NoOptions';
            error( errorStruct );
        end

        % multiple states{ index_data } / single options{ index_data }
        if ~isscalar( states{ index_data } ) && isscalar( options{ index_data } )
            options{ index_data } = repmat( options{ index_data }, size( states{ index_data } ) );
        end

        % ensure equal number of dimensions and sizes
        auxiliary.mustBeEqualSize( states{ index_data }, options{ index_data } );

        %------------------------------------------------------------------
        % b) precompute matrix for linear LSE
        %------------------------------------------------------------------
        % coefficient matrix for linear LSE
        % X = [ sum( xdc_array( index_data ).positions_ctr.^2, 2 ) / physical_values.squaremeter, xdc_array( index_data ).positions_ctr / physical_values.meter, ones( xdc_array( index_data ).N_elements, 1 ) ];
        X = [ sum( xdc_array( index_data ).positions_ctr( :, 1 ).^2, 2 ) / physical_values.squaremeter, xdc_array( index_data ).positions_ctr( :, 1 ) / physical_values.meter, ones( xdc_array( index_data ).N_elements, 1 ) ];

        %------------------------------------------------------------------
        % c) predict times-of-flight for all states
        %------------------------------------------------------------------
        times_of_flight_init = calibration.function_tof( xdc_array( index_data ).positions_ctr, states{ index_data }, [ options{ index_data }.lens_thickness ]', [ options{ index_data }.c_lens ]' );

        % ensure cell array for times_of_flight_init
        if ~iscell( times_of_flight_init )
            times_of_flight_init = { times_of_flight_init };
        end

        %------------------------------------------------------------------
        % d) estimate speed of sound for each point-like target
        %------------------------------------------------------------------
        % number of point-like targets
        N_targets = numel( states{ index_data } );

        tof_ctr = cell( size( states{ index_data } ) );
        intervals_t = cell( size( states{ index_data } ) );

        % statistics of the pulse shape
        pulse_shape{ index_data } = cell( size( states{ index_data } ) );
        pulse_shape_mean{ index_data } = cell( size( states{ index_data } ) );
        pulse_shape_std_dev{ index_data } = cell( size( states{ index_data } ) );

        % results for linear LSE
        pos_r0_linear = physical_values.meter( zeros( N_targets, 2 ) );
        c_avg_linear = physical_values.meter_per_second( zeros( N_targets, 1 ) );

        % results for nonlinear LSE
        pos_r0_nonlinear = physical_values.meter( zeros( N_targets, 3 ) );
        c_avg_nonlinear = physical_values.meter_per_second( zeros( N_targets, 1 ) );

        % estimated times of flight
        times_of_flight_est = cell( size( states{ index_data } ) );

        % relative RMSE of linear LSE
        rel_RMSE{ index_data } = zeros( size( states{ index_data } ) );

        % iterate point-like targets
        for index_target = 1:N_targets

            %--------------------------------------------------------------
            % a) compute time intervals based on predicted times-of-flight and waveform center
            %--------------------------------------------------------------
            % predict waveform centers using times-of-flight
            tof_ctr{ index_target } = times_of_flight_init{ index_target }( options{ index_data }( index_target ).indices_elements ) + options{ index_data }( index_target ).time_shift_ctr;

            % create time intervals
            intervals_t{ index_target } = move( options{ index_data }( index_target ).interval_window_t, tof_ctr{ index_target } );

            %--------------------------------------------------------------
            % b) interpolate QPW data along time axis
            %--------------------------------------------------------------
            u_rx_tilde_qpw_int = interpolate( u_rx_tilde_qpw( index_data ), options{ index_data }( index_target ).factor_interp );

            %--------------------------------------------------------------
            % b) compute inter-element correlation coefficients, lags, and pulse shape
            %--------------------------------------------------------------
            % cut out waveforms (apply windows)
            u_rx_tilde_qpw_int_window = cut_out( u_rx_tilde_qpw_int, [ intervals_t{ index_target }.lb ], [ intervals_t{ index_target }.ub ], num2cell( options{ index_data }( index_target ).indices_elements ), options{ index_data }( index_target ).setting_window );

            % compute correlation coefficients
            xcorr( u_rx_tilde_qpw_int_window );

            % initialize lags with zeros
            lags_adjacent_cut = physical_values.second( zeros( 1, numel( options{ index_data }( index_target ).indices_elements ) ) );
            lags_adjacent = physical_values.second( zeros( 1, numel( options{ index_data }( index_target ).indices_elements ) ) );

            % iterate specified elements
            for index_selected = 2:numel( options{ index_data }( index_target ).indices_elements )

                % extract RF data of adjacent channels
                u_rx_tilde_qpw_int_window_act = u_rx_tilde_qpw_int_window( index_selected );
                u_rx_tilde_qpw_int_window_prev = u_rx_tilde_qpw_int_window( index_selected - 1 );

                % compute inter-element correlation coefficients
                [ data_pw_int_cut_corr, data_pw_int_cut_corr_lags ] = xcorr( u_rx_tilde_qpw_int_window_act.samples / norm( u_rx_tilde_qpw_int_window_act.samples ), u_rx_tilde_qpw_int_window_prev.samples / norm( u_rx_tilde_qpw_int_window_prev.samples ) );

                % detect and save maximum of cross-correlation
                [ ~, index_max ] = max( data_pw_int_cut_corr );

                % estimate relative time delays
                lags_adjacent_cut( index_selected ) = data_pw_int_cut_corr_lags( index_max );
                lags_adjacent( index_selected ) = data_pw_int_cut_corr_lags( index_max ) * u_rx_tilde_qpw_int_window_act.axis.delta + u_rx_tilde_qpw_int_window_act.axis.members( 1 ) - u_rx_tilde_qpw_int_window_prev.axis.members( 1 );

                % illustrate result
%                 figure(999);
%                 plot( u_rx_tilde_qpw_int_window_act.axis.members, u_rx_tilde_qpw_int_window_act.samples / max( u_rx_tilde_qpw_int_window_act.samples ), u_rx_tilde_qpw_int_window_prev.axis.members, u_rx_tilde_qpw_int_window_prev.samples / max( u_rx_tilde_qpw_int_window_prev.samples ) );

                % store pulse shape
%                 if index_selected == 2
%                     pulse_shape{ index_data }{ index_target }( :, 1 ) =
%                     pulse_shape( :, 1 ) = u_rx_tilde_qpw_int_window_prev;
%                 end
%                 pulse_shape( :, index_selected ) = circshift( u_rx_tilde_qpw_int_window_act, -sum( lags_adjacent_cut( 1:index_selected ) ) );

            end % for index_selected = 2:numel( options{ index_data }( index_target ).indices_elements )

            % mean and variance of the pulse shapes
%           pulse_shape_mean{ index_data }{ index_target } = mean( pulse_shape, 2 );
%           pulse_shape_std_dev{ index_data }{ index_target } = sqrt( var( pulse_shape, 0, 2 ) );

            %--------------------------------------------------------------
            % d) estimate lateral position of target
            %--------------------------------------------------------------
            % integrate inter-element delays and find lateral position of minimum
            lags_adjacent_cs = cumsum( lags_adjacent );
            [ lags_adjacent_cs_min, index_selected_min ] = min( lags_adjacent_cs );
            lags_adjacent_cs = lags_adjacent_cs - lags_adjacent_cs_min;

            indicator_min = lags_adjacent_cs == physical_values.second( 0 );
            if sum( indicator_min ) > 1
                index_element_min = round( sum( options{ index_data }( index_target ).indices_elements( indicator_min ) ) / sum( indicator_min ) );
                position_lateral_min = sum( xdc_array( index_data ).positions_ctr( options{ index_data }( index_target ).indices_elements( indicator_min ), : ) ) / sum( indicator_min );
            else
                index_element_min = options{ index_data }( index_target ).indices_elements( index_selected_min );
                position_lateral_min = xdc_array( index_data ).positions_ctr( index_element_min, : );
            end

            %--------------------------------------------------------------
            % e) estimate axial position of target and times-of-flight
            %--------------------------------------------------------------
            % find maximum of envelope of interpolated RF data
            u_rx_tilde_qpw_int_window_env = abs( hilbert( u_rx_tilde_qpw_int_window( index_selected_min ).samples ) );
            [ ~, index_max ] = max( u_rx_tilde_qpw_int_window_env );
            time_max = u_rx_tilde_qpw_int_window( index_selected_min ).axis.members( index_max );

            % estimate axial position of target (linearize estimate)
            time_target = time_max - options{ index_data }( index_target ).time_shift_ctr;

            % estimate times-of-flight
            times_of_flight_est{ index_target } = time_target + lags_adjacent_cs;

            %--------------------------------------------------------------
            % f) linear estimate
            %--------------------------------------------------------------
            % estimate position of the point-like targets and speed of sound (linearized model)
            X_Gram = X( options{ index_data }( index_target ).indices_elements, : )' * X( options{ index_data }( index_target ).indices_elements, : );
            y = ( times_of_flight_est{ index_target }( : ) - time_target / 2 ).^2;
            X_Herm_times_y = X( options{ index_data }( index_target ).indices_elements, : )' * y;

            % solve linear system
            theta = X_Gram \ X_Herm_times_y;

            % extract target position and speed of sound (cf. [1])
            pos_r0_linear( index_target, 1 ) = -0.5 * physical_values.meter * ( theta( 2 ) / theta( 1 ) );
%           pos_r0_linear( index_target, 2 ) = -0.5 * physical_values.meter * theta( 3 ) / theta( 1 );
%           pos_r0_linear( index_target, 2 ) = sqrt( physical_values.squaremeter * theta( 4 ) / theta( 1 ) - pos_r0_linear( index_target, 1 )^2 - pos_r0_linear( index_target, 2 )^2 );
            pos_r0_linear( index_target, 3 ) = sqrt( physical_values.squaremeter * ( theta( end ) / theta( 1 ) ) - pos_r0_linear( index_target, 1 )^2 );
            c_avg_linear( index_target ) = ( 1 - options{ index_data }( index_target ).lens_thickness / pos_r0_linear( index_target, 3 ) ) / ( sqrt( theta( 1 ) ) / physical_values.meter - options{ index_data }( index_target ).lens_thickness / ( pos_r0_linear( index_target, 3 ) * options{ index_data }( index_target ).c_lens ) );

            %--------------------------------------------------------------
            % g) nonlinear estimate
            %--------------------------------------------------------------
            % initial state
            theta_0 = [ double( states{ index_data }( index_target ).position_target ), double( states{ index_data }( index_target ).c_avg ) ];

            % boundaries
            theta_lbs = [ -2e-2, -5e-3, 0,    1450 ];
            theta_ubs = [  2e-2,  5e-3, 8e-2, 1580 ];

            % set optimization options
            options_optimization = optimoptions( 'lsqcurvefit', 'Algorithm', 'trust-region-reflective', 'FunValCheck', 'on', 'Diagnostics', 'on', 'Display', 'iter-detailed', 'FunctionTolerance', 1e-10, 'OptimalityTolerance', 1e-10, 'StepTolerance', 1e-10, 'SpecifyObjectiveGradient', true, 'CheckGradients', false, 'FiniteDifferenceType', 'central', 'FiniteDifferenceStepSize', 1e-10, 'MaxFunctionEvaluations', 5e3, 'MaxIterations', 5e3 );

            % find solution to nonlinear least squares problem
            [ theta_tof, resnorm, residual, exitflag, output ] = lsqcurvefit( @tof_us, theta_0, double( xdc_array( index_data ).positions_ctr ), double( times_of_flight_est{ index_target } ) * 1e6, theta_lbs, theta_ubs, options_optimization );

            % extract target position and speed of sound
            pos_r0_nonlinear( index_target, : ) = physical_values.meter( theta_tof( 1:3 ) );
            c_avg_nonlinear( index_target ) = physical_values.meter_per_second( theta_tof( 4 ) );

            % compute estimation error
            rel_RMSE_0 = norm( double( times_of_flight_est{ index_target } ) * 1e6 - tof_us( theta_0, double( xdc_array( index_data ).positions_ctr ) ), 'fro' ) ./ norm( double( times_of_flight_est{ index_target } ) * 1e6, 'fro' );
            rel_RMSE{ index_data }( index_target ) = norm( double( times_of_flight_est{ index_target } ) * 1e6 - tof_us( theta_tof, double( xdc_array( index_data ).positions_ctr ) ), 'fro' ) ./ norm( double( times_of_flight_est{ index_target } ) * 1e6, 'fro' );

            % check for improvement
            if rel_RMSE{ index_data }( index_target ) > rel_RMSE_0
                warning('No improvement!');
            end

        end % for index_target = 1:N_targets

        %------------------------------------------------------------------
        % 6.) create estimated states and compute errors
        %------------------------------------------------------------------
        % create estimated states
        states{ index_data } = calibration.state( pos_r0_linear, c_avg_linear );
        states_nonlinear{ index_data } = calibration.state( pos_r0_nonlinear, c_avg_nonlinear );

        % display errors
        pos_r0_error = ( pos_r0_nonlinear - pos_r0_linear ) * 1e6
        c_avg_error = ( c_avg_nonlinear - c_avg_linear )

        % predict times-of-flight for estimated states
        times_of_flight_pre = calibration.function_tof( xdc_array( index_data ).positions_ctr, states{ index_data }, [ options{ index_data }.lens_thickness ]', [ options{ index_data }.c_lens ]' );

        % ensure cell array for times_of_flight_pre
        if ~iscell( times_of_flight_pre )
            times_of_flight_pre = { times_of_flight_pre };
        end

        u_rx_tilde_qpw_int_dB = illustration.dB( hilbert( u_rx_tilde_qpw_int.samples ), 20 );

        for index_target = 1:N_targets

            % compute estimation error
            times_of_flight_error = times_of_flight_est{ index_target } - times_of_flight_pre{ index_target }( options{ index_data }( index_target ).indices_elements )';
            rel_RMSE{ index_data }( index_target ) = norm( times_of_flight_error ) / norm( times_of_flight_pre{ index_target }( options{ index_data }( index_target ).indices_elements ) );

            %--------------------------------------------------------------
            % g) illustrate results
            %--------------------------------------------------------------
            dynamic_range_dB = 60;

            figure( index_data );
%         subplot( N_targets, 4, 1 );
%         imagesc( options{ index_data }( index_target ).indices_elements, patches_int( index_target ).indices_axial, pulse_shape );
%         title( 'Aligned Waveforms' );
%         subplot( 3, 2, 2 );
%         plot( (1:N_samples_window_int), pulse_shape_mean{ index_data }{ index_target }, '-b', (1:N_samples_window_int), pulse_shape_mean{ index_data }{ index_target } - pulse_shape_std_dev{ index_data }{ index_target }, '--r', (1:N_samples_window_int), pulse_shape_mean{ index_data }{ index_target } + pulse_shape_std_dev{ index_data }{ index_target }, '--r' );
%         title( 'Mean Pulse Shape & Std. Dev.' );
            subplot( 2, N_targets, index_target );
            imagesc( options{ index_data }( index_target ).indices_elements, double( u_rx_tilde_qpw_int.axis.members ), u_rx_tilde_qpw_int_dB( :, options{ index_data }( index_target ).indices_elements ), [ - dynamic_range_dB, 0 ] );
            line( options{ index_data }( index_target ).indices_elements, double( tof_ctr{ index_target } ), 'Color', [ 1, 1, 0.99 ], 'LineWidth', 2, 'LineStyle', ':' );
            line( options{ index_data }( index_target ).indices_elements, double( [ intervals_t{ index_target }.lb ] ), 'Color', [1,1,0.99], 'LineWidth', 2, 'LineStyle', ':' );
            line( options{ index_data }( index_target ).indices_elements, double( [ intervals_t{ index_target }.ub ] ), 'Color', [1,1,0.99], 'LineWidth', 2, 'LineStyle', ':' );
            line( options{ index_data }( index_target ).indices_elements, times_of_flight_est{ index_target }, 'Color', 'r', 'LineWidth', 3, 'LineStyle', '--' );
            ylim( [ double( min( [ intervals_t{ index_target }.lb ] ) ), double( max( [ intervals_t{ index_target }.ub ] ) ) ] );
            title( 'Envelope of RF Data and Estimated Times-of-Flight' );
            colorbar;
%             subplot( 4, N_targets, N_targets + index_target );
%             plot( options{ index_data }( index_target ).indices_elements, lags_adjacent );
%             title( 'Lags of Adjacent Array Elements' );
%             subplot( 4, N_targets, 2 * N_targets + index_target );
%             plot( options{ index_data }( index_target ).indices_elements, lags_adjacent_cs, index_element_min, 0, 'r+' );
%             title( 'Cumulated Lags and Lateral Target Position' );
            subplot( 2, N_targets, 1 * N_targets + index_target );
            plot( options{ index_data }( index_target ).indices_elements, times_of_flight_est{ index_target }, 'r--',...
                (1:xdc_array( index_data ).N_elements), times_of_flight_init{ index_target }, 'g--',...
                (1:xdc_array( index_data ).N_elements), times_of_flight_pre{ index_target }, 'b' );
            title( 'Estimated, Initial, and Predicted Times-of-Flight' );

        end % for index_target = 1:N_targets

    end % for index_data = 1:numel( u_rx_tilde_qpw )

    % avoid cell arrays for single u_rx_tilde_qpw
    if isscalar( u_rx_tilde_qpw )
        states = states{ 1 };
        rel_RMSE = rel_RMSE{ 1 };
        pulse_shape_mean = pulse_shape_mean{ 1 };
        pulse_shape_std_dev = pulse_shape_std_dev{ 1 };
    end

    % compute TOFs
	function [ y, J ] = tof_us( theta, positions )

        % compute distances
        vect_r0_r = [ positions, zeros( size( positions, 1 ), 1 ) ] - theta( 1:( end - 1 ) );
        dist = vecnorm( vect_r0_r, 2, 2 );

        % compute round-trip times-of-flight (us)
        y = 1e6 * ( theta( end - 1 ) + dist' ) / theta( end );

        % check if Jacobian is required
        if nargout > 1

            % compute Jacobian
            J = zeros( numel( y ), numel( theta ) );

            % partial derivatives w/ respect to lateral positions
            for index_dim = 1:( numel( theta ) - 2 )
                temp = - vect_r0_r( :, index_dim ) ./ dist;
                J( :, index_dim ) = 1e6 * temp( : ) / theta( end );
            end

            % partial derivative w/ respect to axial position
            temp = 1 - vect_r0_r( :, end ) ./ dist;
            J( :, end - 1 ) = 1e6 * temp( : ) / theta( end );

            % partial derivative w/ respect to SoS
            J( :, end ) = - y( : ) / theta( end );

        end % if nargout > 1

    end % function y = tof( theta, positions )

end % function [ states, rel_RMSE, pulse_shape_mean, pulse_shape_std_dev ] = estimate_SOS_point( u_rx_tilde_qpw, xdc_array, states, options )
