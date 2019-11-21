function [ states, rel_RMSE, pulse_shape_mean, pulse_shape_std_dev ] = estimate_SOS_point_qsw( u_rx_tilde_qsw, xdc_array, states, options )
%
% estimate the average speed of sound using
% the inter-element cross-correlations for
% multiple point-like targets (cf. [1])
%
% [1] S. W. Flax and M. O'Donnell, "Phase-Aberration Correction Using Signals From Point Reflectors and Diffuse Scatterers: Basic Principles",
%     IEEE TUFFC, Vol. 35, No. 6, Nov. 1988, pp. 758-767
%
% requires: optimization toolbox
%
% author: Martin F. Schiffner
% date: 2014-09-20
% modified: 2019-11-20

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
        errorStruct.identifier = 'estimate_SOS_point_qsw:NoRegularPlanarArray';
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
	pulse_shape = cell( size( u_rx_tilde_qsw ) );
	pulse_shape_mean = cell( size( u_rx_tilde_qsw ) );
	pulse_shape_std_dev = cell( size( u_rx_tilde_qsw ) );

	% iterate signal matrices
	for index_data = 1:numel( u_rx_tilde_qsw )

        %------------------------------------------------------------------
        % a) check arguments
        %------------------------------------------------------------------
        % ensure class discretizations.signal_matrix
        if ~isa( u_rx_tilde_qsw{ index_data }, 'discretizations.signal_matrix' )
            errorStruct.message = 'u_rx_tilde_qsw must be discretizations.signal_matrix!';
            errorStruct.identifier = 'estimate_SOS_point_qsw:NoSignalMatrices';
            error( errorStruct );
        end

        % ensure valid number of signal matrices
        if numel( u_rx_tilde_qsw{ index_data } ) ~= xdc_array( index_data ).N_elements
            errorStruct.message = sprintf( 'The number of elements in u_rx_tilde_qsw{ %d } must equal the number of elements in xdc_array( %d )!', index_data, index_data );
            errorStruct.identifier = 'estimate_SOS_point_qsw:InvalidNumberOfSignalMatrices';
            error( errorStruct );
        end

        % ensure valid numbers of signals
        if any( [ u_rx_tilde_qsw{ index_data }.N_signals ] ~= xdc_array( index_data ).N_elements )
            errorStruct.message = sprintf( 'The number of signals in u_rx_tilde_qsw( %d ) must equal the number of elements in xdc_array( %d )!', index_data, index_data );
            errorStruct.identifier = 'estimate_SOS_point_qsw:InvalidNumberOfSignals';
            error( errorStruct );
        end

        % ensure class calibration.state
        if ~isa( states{ index_data }, 'calibration.state' )
            errorStruct.message = sprintf( 'states{ %d } must be calibration.state!', index_data );
            errorStruct.identifier = 'estimate_SOS_point_qsw:NoStates';
            error( errorStruct );
        end

        % ensure class calibration.options
        if ~isa( options{ index_data }, 'calibration.options' )
            errorStruct.message = sprintf( 'options{ %d } must be calibration.options!', index_data );
            errorStruct.identifier = 'estimate_SOS_point_qsw:NoOptions';
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
        times_of_flight_init = calibration.function_tof_qsw( xdc_array( index_data ).positions_ctr, states{ index_data } );

        % ensure cell array for times_of_flight_init
        if ~iscell( times_of_flight_init )
            times_of_flight_init = { times_of_flight_init };
        end

        %------------------------------------------------------------------
        % c) estimate speed of sound for each point-like target
        %------------------------------------------------------------------
        % number of point-like targets
        N_targets = numel( states{ index_data } );

        tof_ctr = cell( size( states{ index_data } ) );
        intervals_t = cell( size( states{ index_data } ) );

        % statistics of the pulse shape
        pulse_shape{ index_data } = cell( size( states{ index_data } ) );
        pulse_shape_mean{ index_data } = cell( size( states{ index_data } ) );
        pulse_shape_std_dev{ index_data } = cell( size( states{ index_data } ) );

        % results for nonlinear LSE
        pos_r0_est = physical_values.meter( zeros( N_targets, 3 ) );
        c_avg_est = physical_values.meter_per_second( zeros( N_targets, 1 ) );

        % estimated times of flight
        times_of_flight_est = cell( size( states{ index_data } ) );

        % relative RMSE of linear LSE
        rel_RMSE{ index_data } = zeros( size( states{ index_data } ) );

        % iterate point-like targets
        for index_target = 1:N_targets

            %--------------------------------------------------------------
            % a) compute time intervals based on predicted TOFs and waveform center
            %--------------------------------------------------------------
            % predict waveform centers using TOFs
            tof_ctr{ index_target } = times_of_flight_init{ index_target }( options{ index_data }( index_target ).indices_elements, options{ index_data }( index_target ).indices_elements ) + options{ index_data }( index_target ).time_shift_ctr;

            % create time intervals
            intervals_t{ index_target } = move( options{ index_data }( index_target ).interval_window_t, tof_ctr{ index_target } );

            %--------------------------------------------------------------
            % b) compute inter-element correlation coefficients and lags
            %--------------------------------------------------------------
            % specify cell array for times_of_flight_est
            lags_adjacent = cell( size( options{ index_data }( index_target ).indices_elements ) );
            lags_adjacent_cs = cell( size( options{ index_data }( index_target ).indices_elements ) );
            times_of_flight_est{ index_target } = cell( size( options{ index_data }( index_target ).indices_elements ) );

            % iterate specified tx elements
            INDICES_TX = options{ index_data }( index_target ).indices_elements;
            for index_selected_tx = 1:numel( INDICES_TX )

                % index of the array element
                index_element_tx = INDICES_TX( index_selected_tx );

                %----------------------------------------------------------
                % a) interpolate SA data and cut out waveforms
                %----------------------------------------------------------
                % interpolate QSW data along the time axis
                u_rx_tilde_qsw_int = interpolate( u_rx_tilde_qsw{ index_data }( index_element_tx ), options{ index_data }( index_target ).factor_interp );

                % cut out waveforms (apply windows)
                u_rx_tilde_qsw_int_window = cut_out( u_rx_tilde_qsw_int, [ intervals_t{ index_target }( index_element_tx, : ).lb ], [ intervals_t{ index_target }( index_element_tx, : ).ub ], num2cell( options{ index_data }( index_target ).indices_elements ), options{ index_data }( index_target ).setting_window );

                %----------------------------------------------------------
                % b) compute inter-element lags
                %----------------------------------------------------------
                % initialize lags with zeros
                lags_adjacent{ index_selected_tx } = physical_values.second( zeros( 1, numel( options{ index_data }( index_target ).indices_elements ) ) );

                % iterate specified rx elements
                for index_selected_rx = 2:numel( options{ index_data }( index_target ).indices_elements )

                    % extract RF data of adjacent channels
                    u_rx_tilde_qsw_int_window_act = u_rx_tilde_qsw_int_window( index_selected_rx );
                    u_rx_tilde_qsw_int_window_prev = u_rx_tilde_qsw_int_window( index_selected_rx - 1 );

                    % compute inter-element correlation coefficients
                    [ data_pw_int_cut_corr, data_pw_int_cut_corr_lags ] = xcorr( u_rx_tilde_qsw_int_window_act.samples / norm( u_rx_tilde_qsw_int_window_act.samples ), u_rx_tilde_qsw_int_window_prev.samples / norm( u_rx_tilde_qsw_int_window_prev.samples ) );

                    % detect and save maximum of cross-correlation
                    [ ~, index_max ] = max( data_pw_int_cut_corr );

                    % estimate relative time delays
                    lags_adjacent{ index_selected_tx }( index_selected_rx ) = data_pw_int_cut_corr_lags( index_max ) * u_rx_tilde_qsw_int_window_act.axis.delta + u_rx_tilde_qsw_int_window_act.axis.members( 1 ) - u_rx_tilde_qsw_int_window_prev.axis.members( 1 );

                    % illustrate result
%                     figure(999);
%                     plot( u_rx_tilde_qsw_int_window_act.axis.members - lags_adjacent{ index_selected_tx }( index_selected_rx ), u_rx_tilde_qsw_int_window_act.samples / max( u_rx_tilde_qsw_int_window_act.samples ), u_rx_tilde_qsw_int_window_prev.axis.members, u_rx_tilde_qsw_int_window_prev.samples / max( u_rx_tilde_qsw_int_window_prev.samples ) );

                end % for index_selected_rx = 2:numel( options{ index_data }( index_target ).indices_elements )

                % integrate inter-element delays and find lateral position of minimum
                lags_adjacent_cs{ index_selected_tx } = cumsum( lags_adjacent{ index_selected_tx }, 2 );
                [ lags_adjacent_cs_min, index_min ] = min( lags_adjacent_cs{ index_selected_tx }, [], 2 );
                lags_adjacent_cs{ index_selected_tx } = lags_adjacent_cs{ index_selected_tx } - lags_adjacent_cs_min;

                % find maximum of envelope of interpolated RF data
                u_rx_tilde_qsw_int_window_env = abs( hilbert( u_rx_tilde_qsw_int_window( index_min ).samples ) );
                [ ~, index_max ] = max( u_rx_tilde_qsw_int_window_env );
                time_max = u_rx_tilde_qsw_int_window( index_min ).axis.members( index_max );

                % estimate minimum time-of-flight
                tof_min = time_max - options{ index_data }( index_target ).time_shift_ctr;

                % estimate TOFs for all rx elements
                times_of_flight_est{ index_target }{ index_selected_tx } = tof_min + lags_adjacent_cs{ index_selected_tx };

            end % for index_selected_tx = 1:numel( options{ index_data }( index_target ).indices_elements )

            % concatenate cell arrays into matrices
            lags_adjacent = cat( 1, lags_adjacent{ : } );
            lags_adjacent_cs = cat( 1, lags_adjacent_cs{ : } );
            times_of_flight_est{ index_target } = cat( 1, times_of_flight_est{ index_target }{ : } );

            %--------------------------------------------------------------
            % c) nonlinear estimate
            %--------------------------------------------------------------
            % initial state
            theta_0 = [ double( states{ index_data }( index_target ).position_target ), double( states{ index_data }( index_target ).c_avg ) ];

            % boundaries
            theta_lbs = [ -2e-2, -5e-3, 0,    1450 ];
            theta_ubs = [  2e-2,  5e-3, 8e-2, 1580 ];

            % set optimization options
            options_optimization = optimoptions( 'lsqcurvefit', 'Algorithm', 'trust-region-reflective', 'FunValCheck', 'on', 'Diagnostics', 'on', 'Display', 'iter-detailed', 'FunctionTolerance', 1e-10, 'OptimalityTolerance', 1e-10, 'StepTolerance', 1e-10, 'SpecifyObjectiveGradient', true, 'CheckGradients', false, 'FiniteDifferenceType', 'central', 'FiniteDifferenceStepSize', 1e-10, 'MaxFunctionEvaluations', 5e3, 'MaxIterations', 5e3 );

            % find solutions to nonlinear least squares problems
%             [ theta_lags, resnorm, residual, exitflag, output ] = lsqcurvefit( @lags_us, theta_0, double( xdc_array( index_data ).positions_ctr ), double( lags_adjacent( :, 2:end ) ) * 1e6, theta_lbs, theta_ubs, options_optimization );
%             [ theta_lags_cs, resnorm, residual, exitflag, output ] = lsqcurvefit( @lags_cs_us, theta_0, double( xdc_array( index_data ).positions_ctr ), double( lags_adjacent_cs ) * 1e6, theta_lbs, theta_ubs, options_optimization );
            [ theta_tof, resnorm, residual, exitflag, output ] = lsqcurvefit( @tof_us, theta_0, double( xdc_array( index_data ).positions_ctr ), double( times_of_flight_est{ index_target } ) * 1e6, theta_lbs, theta_ubs, options_optimization );

            % extract target position and speed of sound
            pos_r0_est( index_target, : ) = physical_values.meter( theta_tof( 1:3 ) );
            c_avg_est( index_target ) = physical_values.meter_per_second( theta_tof( 4 ) );

            % compute estimation error
            rel_RMSE_0 = norm( double( times_of_flight_est{ index_target } ) * 1e6 - tof_us( theta_0, double( xdc_array( index_data ).positions_ctr ) ), 'fro' ) ./ norm( double( times_of_flight_est{ index_target } ) * 1e6, 'fro' );
            rel_RMSE{ index_data }( index_target ) = norm( double( times_of_flight_est{ index_target } ) * 1e6 - tof_us( theta_tof, double( xdc_array( index_data ).positions_ctr ) ), 'fro' ) ./ norm( double( times_of_flight_est{ index_target } ) * 1e6, 'fro' );

            % check for improvement
            if rel_RMSE{ index_data }( index_target ) > rel_RMSE_0
                warning('No improvement!');
            end

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
        pulse_shape = pulse_shape{ 1 };
        pulse_shape_mean = pulse_shape_mean{ 1 };
        pulse_shape_std_dev = pulse_shape_std_dev{ 1 };
    end

	% compute differences of TOFs for adjacent elements
	function [ y, J ] = lags_us( theta, positions )

        % compute distances
        vect_r0_r = [ positions, zeros( size( positions, 1 ), 1 ) ] - theta( 1:( end - 1 ) );
        dist = vecnorm( vect_r0_r, 2, 2 );

        % compute differences of TOFs
        y = repmat( 1e6 * diff( dist' ) / theta( end ), [ numel( INDICES_TX ), 1 ] );

        % check if Jacobian is required
        if nargout > 1

            % compute Jacobian
            J = zeros( numel( y ), numel( theta ) );

            % partial derivatives w/ respect to position
            for index_dim = 1:( numel( theta ) - 1 )
                temp = - vect_r0_r( :, index_dim ) ./ dist;
                temp = repmat( diff( temp' ), [ numel( INDICES_TX ), 1 ] );
                J( :, index_dim ) = 1e6 * temp( : ) / theta( end );
            end

            % partial derivative w/ respect to SoS
            J( :, end ) = - y( : ) / theta( end );

        end

	end % function [ y, J ] = lags_us( theta, positions )

    % compute differences of TOFs w/ respect to first element
	function [ y, J ] = lags_cs_us( theta, positions )

        % compute distances
        vect_r0_r = [ positions, zeros( size( positions, 1 ), 1 ) ] - theta( 1:( end - 1 ) );
        dist = vecnorm( vect_r0_r, 2, 2 );

        % compute differences of TOFs
        y = repmat( 1e6 * ( dist' - dist( 1 ) ) / theta( end ), [ numel( INDICES_TX ), 1 ] );

        % check if Jacobian is required
        if nargout > 1

            % compute Jacobian
            J = zeros( numel( y ), numel( theta ) );

            % partial derivatives w/ respect to position
            for index_dim = 1:( numel( theta ) - 1 )
                temp = - vect_r0_r( :, index_dim ) ./ dist;
                temp = repmat( temp' - temp( 1 ), [ numel( INDICES_TX ), 1 ] );
                J( :, index_dim ) = 1e6 * temp( : ) / theta( end );
            end

            % partial derivative w/ respect to SoS
            J( :, end ) = - y( : ) / theta( end );

        end

	end % function [ y, J ] = lags_cs_us( theta, positions )

	% compute TOFs
	function [ y, J ] = tof_us( theta, positions )

        % compute distances
        vect_r0_r = [ positions, zeros( size( positions, 1 ), 1 ) ] - theta( 1:( end - 1 ) );
        dist = vecnorm( vect_r0_r, 2, 2 );

        % compute round-trip times-of-flight (us)
        y = ( dist + dist' ) / theta( end );
        y = 1e6 * y( INDICES_TX, : );

        % check if Jacobian is required
        if nargout > 1

            % compute Jacobian
            J = zeros( numel( y ), numel( theta ) );

            % partial derivatives w/ respect to position
            for index_dim = 1:( numel( theta ) - 1 )
                temp = - vect_r0_r( :, index_dim ) ./ dist;
                temp = temp( INDICES_TX ) + temp';
                J( :, index_dim ) = 1e6 * temp( : ) / theta( end );
            end

            % partial derivative w/ respect to SoS
            J( :, end ) = - y( : ) / theta( end );

        end % if nargout > 1

    end % function y = tof( theta, positions )

end % function [ states, rel_RMSE, pulse_shape_mean, pulse_shape_std_dev ] = estimate_SOS_point_qsw( u_rx_tilde_qsw, xdc_array, states, options )
