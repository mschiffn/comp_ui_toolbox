function [ states, rel_RMSE ] = estimate_sos_point_qsw( u_rx_tilde_qsw, xdc_array, states, options )
%
% Estimates the average speed of sound and the positions of
% multiple point-like reflectors.
%
% The function analyzes
% the times-of-flight of
% the induced echoes using
% the inter-element cross-correlations
% (see [1], [2]).
%
% INPUT:
%   u_rx_tilde_qsw = RF voltage signals obtained by complete SA sequence
%   xdc_array = transducer array
%   states = initial values of the average speed of sound and position of the ROI
%   options = calibration.options.sos_qsw
%
% OUTPUT:
%   states = corrected input states
%   rel_RMSE = relative root mean-squared fitting error
%
% REQUIREMENTS:
%	- MATLAB Optimization Toolbox
%
% REFERENCES:
%	[1] S. W. Flax and M. O'Donnell, "Phase-Aberration Correction Using Signals From Point Reflectors and Diffuse Scatterers: Basic Principles",
%       IEEE TUFFC, Vol. 35, No. 6, Nov. 1988, pp. 758-767
%       DOI: 10.1109/58.9333
%	[2] M. E. Anderson and G. E. Trahey, "The direct estimation of sound speed using pulse–echo ultrasound",
%       J. Acoust. Soc. Am., Vol. 104, No. 5, Nov. 1998, pp. 3099-3106
%       DOI: 10.1121/1.423889
%
% author: Martin F. Schiffner
% date: 2014-09-20
% modified: 2020-02-04

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
        errorStruct.identifier = 'estimate_sos_point_qsw:NoRegularPlanarArray';
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
            errorStruct.identifier = 'estimate_sos_point_qsw:NoSignalMatrices';
            error( errorStruct );
        end

        % ensure valid number of signal matrices
        if numel( u_rx_tilde_qsw{ index_data } ) ~= xdc_array( index_data ).N_elements
            errorStruct.message = sprintf( 'The number of elements in u_rx_tilde_qsw{ %d } must equal the number of elements in xdc_array( %d )!', index_data, index_data );
            errorStruct.identifier = 'estimate_sos_point_qsw:InvalidNumberOfSignalMatrices';
            error( errorStruct );
        end

        % ensure valid numbers of signals
        if any( [ u_rx_tilde_qsw{ index_data }.N_signals ] ~= xdc_array( index_data ).N_elements )
            errorStruct.message = sprintf( 'The number of signals in u_rx_tilde_qsw( %d ) must equal the number of elements in xdc_array( %d )!', index_data, index_data );
            errorStruct.identifier = 'estimate_sos_point_qsw:InvalidNumberOfSignals';
            error( errorStruct );
        end

        % ensure class calibration.state
        if ~isa( states{ index_data }, 'calibration.state' )
            errorStruct.message = sprintf( 'states{ %d } must be calibration.state!', index_data );
            errorStruct.identifier = 'estimate_sos_point_qsw:NoStates';
            error( errorStruct );
        end

        % ensure class calibration.options.sos_qsw
        if ~isa( options{ index_data }, 'calibration.options.sos_qsw' )
            errorStruct.message = sprintf( 'options{ %d } must be calibration.options.sos_qsw!', index_data );
            errorStruct.identifier = 'estimate_sos_point_qsw:NoOptionsSoSQSW';
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
            % indices of tx / rx elements
            INDICES_TX = options{ index_data }( index_target ).indices_elements_tx;
            INDICES_RX = options{ index_data }( index_target ).indices_elements_rx;

            % predict waveform centers using TOFs
            tof_ctr{ index_target } = times_of_flight_init{ index_target }( INDICES_TX, INDICES_RX ) + options{ index_data }( index_target ).time_shift_ctr;

            % create time intervals
            intervals_t{ index_target } = move( options{ index_data }( index_target ).interval_window_t, tof_ctr{ index_target } );

            %--------------------------------------------------------------
            % b) compute inter-element correlation coefficients and lags
            %--------------------------------------------------------------
            % specify cell array for times_of_flight_est
            lags_adjacent = cell( size( INDICES_TX ) );
            lags_adjacent_cs = cell( size( INDICES_TX ) );
            times_of_flight_est{ index_target } = cell( size( INDICES_TX ) );

            % iterate specified tx elements
            for index_selected_tx = 1:numel( INDICES_TX )

                % index of the array element
                index_element_tx = INDICES_TX( index_selected_tx );

                %----------------------------------------------------------
                % a) interpolate SA data and cut out waveforms
                %----------------------------------------------------------
                % interpolate QSW data along the time axis
                u_rx_tilde_qsw_int = interpolate( u_rx_tilde_qsw{ index_data }( index_element_tx ), options{ index_data }( index_target ).factor_interp );

                % cut out waveforms (apply windows)
                u_rx_tilde_qsw_int_window = cut_out( u_rx_tilde_qsw_int, cat( 1, intervals_t{ index_target }( index_selected_tx, : ).lb ), cat( 1, intervals_t{ index_target }( index_selected_tx, : ).ub ), num2cell( INDICES_RX ), options{ index_data }( index_target ).setting_window );

                % illustrate cut out
                if options{ index_data }( index_target ).display

                    figure( 998 );
                    imagesc( INDICES_RX, double( u_rx_tilde_qsw_int.axis.members ), illustration.dB( hilbert( u_rx_tilde_qsw_int.samples( :, INDICES_RX ) ), 20 ), [ -60, 0 ] );
                    line( INDICES_RX, double( [ intervals_t{ index_target }( index_selected_tx, : ).lb ] ), 'Color', [1,1,0.99], 'LineWidth', 2, 'LineStyle', ':' );
                    line( INDICES_RX, double( [ intervals_t{ index_target }( index_selected_tx, : ).ub ] ), 'Color', [1,1,0.99], 'LineWidth', 2, 'LineStyle', ':' );

                end % if options{ index_data }( index_target ).display

                %----------------------------------------------------------
                % b) compute inter-element lags
                %----------------------------------------------------------
                u_rx_tilde_qsw_int_window = processing.signal( cat( 1, u_rx_tilde_qsw_int_window.axis ), { u_rx_tilde_qsw_int_window.samples }' );
                [ ~, lags_adjacent{ index_selected_tx } ] = xcorr_max( u_rx_tilde_qsw_int_window );

                %----------------------------------------------------------
                % c) estimate TOFs
                %----------------------------------------------------------
                % integrate inter-element delays and find lateral position of minimum
                lags_adjacent_cs{ index_selected_tx } = cumsum( lags_adjacent{ index_selected_tx }, 1 );
                [ lags_adjacent_cs_min, index_min ] = min( lags_adjacent_cs{ index_selected_tx }, [], 1 );
                lags_adjacent_cs{ index_selected_tx } = lags_adjacent_cs{ index_selected_tx } - lags_adjacent_cs_min;

                % find maximum of envelope of interpolated RF data
                u_rx_tilde_qsw_int_window_env = abs( hilbert( u_rx_tilde_qsw_int_window( index_min ).samples ) );
                [ ~, index_max ] = max( u_rx_tilde_qsw_int_window_env );
                time_max = u_rx_tilde_qsw_int_window( index_min ).axis.members( index_max );

                % estimate minimum time-of-flight
                tof_min = time_max - options{ index_data }( index_target ).time_shift_ctr;

                % estimate TOFs for all rx elements
                times_of_flight_est{ index_target }{ index_selected_tx } = tof_min + lags_adjacent_cs{ index_selected_tx }';

            end % for index_selected_tx = 1:numel( INDICES_TX )

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
            theta_lbs = [ -2e-2, -5e-3, 0,    1400 ];
            theta_ubs = [  2e-2,  5e-3, 8e-2, 1850 ];

            % set optimization options
            options_optimization = optimoptions( 'lsqcurvefit', 'Algorithm', 'trust-region-reflective', 'FunValCheck', 'on', 'Diagnostics', 'on', 'Display', 'iter-detailed', 'FunctionTolerance', 1e-10, 'OptimalityTolerance', 1e-10, 'StepTolerance', 1e-10, 'SpecifyObjectiveGradient', true, 'CheckGradients', false, 'FiniteDifferenceType', 'central', 'FiniteDifferenceStepSize', 1e-10, 'MaxFunctionEvaluations', 5e3, 'MaxIterations', 5e3 );

            % find solutions to nonlinear least squares problems
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

            %--------------------------------------------------------------
            % d) illustration
            %--------------------------------------------------------------
            figure( index_data );
            subplot( 3, N_targets, index_target );
            imagesc( double( times_of_flight_est{ index_target } ) );
            title( 'Detected' );
            subplot( 3, N_targets, N_targets + index_target );
            imagesc( double( tof_us( theta_tof, double( xdc_array( index_data ).positions_ctr ) ) ) );
            title( 'Estimated' );
            subplot( 3, N_targets, 2 * N_targets + index_target );
            imagesc( double( times_of_flight_est{ index_target } ) * 1e6 - tof_us( theta_tof, double( xdc_array( index_data ).positions_ctr ) ) );
            title( sprintf( 'Error (%.2f %%)', rel_RMSE{ index_data }( index_target ) * 1e2 ) );

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
        y = ( dist + dist' ) / theta( end );
        y = 1e6 * y( INDICES_TX, INDICES_RX );

        % check if Jacobian is required
        if nargout > 1

            % compute Jacobian
            J = zeros( numel( y ), numel( theta ) );

            % partial derivatives w/ respect to position
            for index_dim = 1:( numel( theta ) - 1 )
                temp = - vect_r0_r( :, index_dim ) ./ dist;
                temp = temp( INDICES_TX ) + temp( INDICES_RX )';
                J( :, index_dim ) = 1e6 * temp( : ) / theta( end );
            end

            % partial derivative w/ respect to SoS
            J( :, end ) = - y( : ) / theta( end );

        end % if nargout > 1

	end % function [ y, J ] = tof_us( theta, positions )

end % function [ states, rel_RMSE ] = estimate_sos_point_qsw( u_rx_tilde_qsw, xdc_array, states, options )
