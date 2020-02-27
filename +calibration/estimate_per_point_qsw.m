function [ e_B_tilde_ref, cal_tx_tilde, cal_rx_tilde, rel_RMSE_local, e_B_tilde, e_B_tilde_mean, e_B_tilde_std_dev ] = estimate_per_point_qsw( u_SA_tilde, xdc_array, states, options )
%
% estimate the pulse-echo responses for
% multiple point-like targets (cf. [1], [2])
%
% [1] L. W. Schmerr Jr., "Fundamentals of Ultrasonic Phased Arrays", 1st, ser. Solid Mechanics and Its Applications.
%     Springer International Publishing, 2015, vol. 215. (doi: 10.1007/978-3-319-07272-2)
% [2] R. Huang and L. W. Schmerr Jr., "Characterization of the system functions of ultrasonic linear phased array inspection systems",
%     Ultrasonics, Vol. 49, 2009, pp. 219-225 (doi: 10.1016/j.ultras.2008.08.004)
%
% author: Martin F. Schiffner
% date: 2019-10-24
% modified: 2020-02-03

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure cell array for u_SA_tilde
	if ~iscell( u_SA_tilde )
        u_SA_tilde = { u_SA_tilde };
	end

	% ensure class scattering.sequences.setups.transducers.array_planar_regular
	if ~isa( xdc_array, 'scattering.sequences.setups.transducers.array_planar_regular' )
        errorStruct.message = 'xdc_array must be scattering.sequences.setups.transducers.array_planar_regular!';
        errorStruct.identifier = 'estimate_per_point_qsw:NoRegularPlanarArray';
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

	% multiple u_SA_tilde / single xdc_array
	if ~isscalar( u_SA_tilde ) && isscalar( xdc_array )
        xdc_array = repmat( xdc_array, size( u_SA_tilde ) );
    end

    % multiple u_SA_tilde / single states
	if ~isscalar( u_SA_tilde ) && isscalar( states )
        states = repmat( states, size( u_SA_tilde ) );
    end

	% multiple u_SA_tilde / single options
	if ~isscalar( u_SA_tilde ) && isscalar( options )
        options = repmat( options, size( u_SA_tilde ) );
    end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( u_SA_tilde, xdc_array, states, options );

    %----------------------------------------------------------------------
	% 2.) check arguments
	%----------------------------------------------------------------------
	% specify cell arrays
	e_B_tilde_ref = cell( size( u_SA_tilde ) );
	cal_tx_tilde = cell( size( u_SA_tilde ) );
	cal_rx_tilde = cell( size( u_SA_tilde ) );
	rel_RMSE_local = cell( size( u_SA_tilde ) );
	e_B_tilde = cell( size( u_SA_tilde ) );
	e_B_tilde_mean = cell( size( u_SA_tilde ) );
	e_B_tilde_std_dev = cell( size( u_SA_tilde ) );

	% iterate SA datasets
	for index_data = 1:numel( u_SA_tilde )

        %------------------------------------------------------------------
        % a) check arguments
        %------------------------------------------------------------------
        % ensure class processing.signal_matrix
        if ~isa( u_SA_tilde{ index_data }, 'processing.signal_matrix' )
            errorStruct.message = sprintf( 'u_SA_tilde{ %d } must be processing.signal_matrix!', index_data );
            errorStruct.identifier = 'estimate_per_point_qsw:NoSignalMatrices';
            error( errorStruct );
        end
% TODO: same delta!
        % ensure valid number of signal matrices
        if numel( u_SA_tilde{ index_data } ) ~= xdc_array( index_data ).N_elements
            errorStruct.message = sprintf( 'The number of elements in u_SA_tilde{ %d } must equal the number of elements in xdc_array( %d )!', index_data, index_data );
            errorStruct.identifier = 'estimate_per_point_qsw:InvalidNumberOfSignalMatrices';
            error( errorStruct );
        end

        % ensure valid numbers of signals
        if any( [ u_SA_tilde{ index_data }.N_signals ] ~= xdc_array( index_data ).N_elements )
            errorStruct.message = sprintf( 'The number of signals in u_SA_tilde{ %d } must equal the number of elements in xdc_array( %d )!', index_data, index_data );
            errorStruct.identifier = 'estimate_per_point_qsw:InvalidNumberOfSignals';
            error( errorStruct );
        end

        % ensure class calibration.state
        if ~isa( states{ index_data }, 'calibration.state' )
            errorStruct.message = 'states{ index_data } must be calibration.state!';
            errorStruct.identifier = 'estimate_per_point_qsw:NoStates';
            error( errorStruct );
        end

        % ensure class calibration.options.per_qsw
        if ~isa( options{ index_data }, 'calibration.options.per_qsw' )
            errorStruct.message = 'options{ index_data } must be calibration.options.per_qsw!';
            errorStruct.identifier = 'estimate_per_point_qsw:NoOptionsPERQSW';
            error( errorStruct );
        end

        % multiple states{ index_data } / single options{ index_data }
        if ~isscalar( states{ index_data } ) && isscalar( options{ index_data } )
            options{ index_data } = repmat( options{ index_data }, size( states{ index_data } ) );
        end

        % ensure equal number of dimensions and sizes
        auxiliary.mustBeEqualSize( states{ index_data }, options{ index_data } );

        %------------------------------------------------------------------
        % b) predict times-of-flight for all states
        %------------------------------------------------------------------
        times_of_flight = calibration.function_tof_qsw( xdc_array( index_data ).positions_ctr, states{ index_data } );

        % ensure cell array for times_of_flight
        if ~iscell( times_of_flight )
            times_of_flight = { times_of_flight };
        end

        %------------------------------------------------------------------
        % c) estimate pulse-echo responses for each point-like target
        %------------------------------------------------------------------
        % number of point-like targets
        N_targets = numel( states{ index_data } );

        % specify cell arrays
        e_B_tilde_ref{ index_data } = cell( size( states{ index_data } ) );
        cal_tx_tilde{ index_data } = cell( size( states{ index_data } ) );
        cal_rx_tilde{ index_data } = cell( size( states{ index_data } ) );
        rel_RMSE_local{ index_data } = cell( size( states{ index_data } ) );
        e_B_tilde{ index_data } = cell( size( states{ index_data } ) );
        e_B_tilde_mean{ index_data } = cell( size( states{ index_data } ) );
        e_B_tilde_std_dev{ index_data } = cell( size( states{ index_data } ) );

        % iterate point-like targets
        for index_target = 1:N_targets

            %--------------------------------------------------------------
            % a) compute time intervals based on predicted times-of-flight and waveform center
            %--------------------------------------------------------------
            % predict waveform starts using times-of-flight
            tof_start = times_of_flight{ index_target }( options{ index_data }( index_target ).indices_elements_tx, options{ index_data }( index_target ).indices_elements_rx );

            % create time intervals for all active elements
            intervals_t = move( options{ index_data }( index_target ).interval_window_t, tof_start + abs( options{ index_data }( index_target ).interval_window_t ) / 2 );

            %--------------------------------------------------------------
            % b) cut out waveforms
            %--------------------------------------------------------------
            % specify cell array for u_SA_tilde_window
            u_SA_tilde_window = cell( size( options{ index_data }( index_target ).indices_elements_tx ) );

            % iterate specified elements
            for index_selected = 1:numel( options{ index_data }( index_target ).indices_elements_tx )

                % index of the array element
                index_element = options{ index_data }( index_target ).indices_elements_tx( index_selected );

                % cut out waveforms (apply windows)
                u_SA_tilde_window{ index_selected } = cut_out( u_SA_tilde{ index_data }( index_element ), [ intervals_t( index_selected, : ).lb ], [ intervals_t( index_selected, : ).ub ], num2cell( options{ index_data }( index_target ).indices_elements_rx ), options{ index_data }( index_target ).setting_window );

            end

            % concatenate waveforms
            u_SA_tilde_window = cat( 1, u_SA_tilde_window{ : } );

            %--------------------------------------------------------------
            % c) initialize pulse-echo measurement setup and ToFs
            %--------------------------------------------------------------
            % spatial discretization options
            method_faces = options{ index_data }( index_target ).method_faces;
            method_FOV = scattering.sequences.setups.discretizations.methods.grid_numbers( ones( numel( states{ index_data }( index_target ).position_target ), 1 ) );
            options_disc_spatial = scattering.sequences.setups.discretizations.options( method_faces, method_FOV );

            % create homogeneous fluid
            rho_0 = physical_values.kilogram_per_cubicmeter( 1000 );
            absorption_model = options{ index_data }( index_target ).handle_absorption_model( states{ index_data }( index_target ).c_avg );
            homogeneous_fluid = scattering.sequences.setups.materials.homogeneous_fluid( rho_0, absorption_model, states{ index_data }( index_target ).c_avg );

            % create field-of-view around point-like target
            FOV_interval_ref = math.interval( physical_values.meter( 0 ), physical_values.meter( 5e-5 ) );
            FOV_intervals = num2cell( move( FOV_interval_ref, states{ index_data }( index_target ).position_target ) );
            FOV = scattering.sequences.setups.fields_of_view.orthotope( FOV_intervals{ : } );

            % create and discretize pulse-echo measurement setup
            setup = scattering.sequences.setups.setup( xdc_array( index_data ), homogeneous_fluid, FOV, 'calibration' );
            setup = discretize( setup, options_disc_spatial );

            % predict waveform starts using times-of-flight
            t_tof_lbs_min = min( min( reshape( [ setup.intervals_tof.lb ], [ setup.xdc_array.N_elements, setup.xdc_array.N_elements ] ), [], 1 ), [], 2 );
            t_tof_ubs_max = max( max( reshape( [ setup.intervals_tof.ub ], [ setup.xdc_array.N_elements, setup.xdc_array.N_elements ] ), [], 1 ), [], 2 );

            %--------------------------------------------------------------
            % d) compute Fourier coefficients and spatial transfer functions
            %--------------------------------------------------------------
            % compute Fourier coefficients (analysis interval is tofs plus pulse length)
            interval_t_tof_quantized = quantize( math.interval( t_tof_lbs_min, t_tof_ubs_max + options{ index_data }( index_target ).interval_window_t.ub ), u_SA_tilde{ index_data }( 1 ).axis.delta );
            u_SA_window = fourier_coefficients( u_SA_tilde_window, abs( interval_t_tof_quantized ), options{ index_data }( index_target ).interval_f );
            u_SA_window = merge( u_SA_window.' );

            % compute spatial transfer functions of the individual vibrating faces
% TODO: anti-aliasing?
            h_transfer = transfer_function( setup, u_SA_window.axis, ( 1:setup.xdc_array.N_elements ) );

            % compute prefactors
            prefactors = compute_prefactors( setup, u_SA_window.axis );

            %--------------------------------------------------------------
            % e) estimate pulse-echo responses
            %--------------------------------------------------------------
            e_B_samples = u_SA_window.samples;

            % iterate specified elements
            for index_selected_tx = 1:numel( options{ index_data }( index_target ).indices_elements_tx )

                % index of the array element
                index_element_tx = options{ index_data }( index_target ).indices_elements_tx( index_selected_tx );

                % extract spatial transmit function
                h_tx = h_transfer( index_element_tx ).samples;

                % iterate specified elements
                for index_selected_rx = 1:numel( options{ index_data }( index_target ).indices_elements_rx )

                    % index of the array element
                    index_element_rx = options{ index_data }( index_target ).indices_elements_rx( index_selected_rx );

                    % total index
                    index = ( index_selected_tx - 1 ) * numel( options{ index_data }( index_target ).indices_elements_rx ) + index_selected_rx;

                    % round-trip transfer function
                    h_rx_tx = double( prefactors.samples .* h_tx .* h_transfer( index_element_rx ).samples );

                    % deconvolve waveform
                    h_rx_tx_abs_squared = abs( h_rx_tx ).^2;
                    h_rx_tx_abs_squared_max = max( h_rx_tx_abs_squared );
                    e_B_samples( :, index ) = u_SA_window.samples( :, index ) .* conj( h_rx_tx ) ./ ( h_rx_tx_abs_squared + options{ index_data }( index_target ).epsilon_squared_PER * h_rx_tx_abs_squared_max );

                    % graphical illustration
%                     figure( 1 );
%                     plot( double( u_SA_window.axis.members ), abs( u_SA_window.samples( :, index ) ), ...
%                           double( u_SA_window.axis.members ), abs( conj( h_rx_tx ) ./ ( h_rx_tx_abs_squared + options{ index_data }( index_target ).epsilon_squared_PER * h_rx_tx_abs_squared_max ) ), ...
%                           double( u_SA_window.axis.members ), abs( e_B_samples( :, index ) ) );

                end % for index_selected_rx = 1:numel( options{ index_data }( index_target ).indices_elements_rx )

            end % for index_selected_tx = 1:numel( options{ index_data }( index_target ).indices_elements_tx )

            % Fourier synthesis
            e_B = processing.signal_matrix( u_SA_window.axis, e_B_samples );
            e_B_tilde{ index_data }{ index_target } = signal( e_B, 0, u_SA_tilde{ index_data }( 1 ).axis.delta );

            % apply windows (use original time window)
            e_B_tilde{ index_data }{ index_target } = cut_out( e_B_tilde{ index_data }{ index_target }, options{ index_data }( index_target ).interval_window_t.lb, options{ index_data }( index_target ).interval_window_t.ub, [], options{ index_data }( index_target ).setting_window );

            % compute mean and standard deviation
            e_B_tilde_mean{ index_data }{ index_target } = processing.signal( e_B_tilde{ index_data }{ index_target }.axis, mean( e_B_tilde{ index_data }{ index_target }.samples, 2 ) );
            e_B_tilde_std_dev{ index_data }{ index_target } = processing.signal( e_B_tilde{ index_data }{ index_target }.axis, sqrt( var( e_B_tilde{ index_data }{ index_target }.samples, [], 2 ) ) );

            % reference pulse-echo response
            index_ref = ( options{ index_data }( index_target ).index_selected_tx_ref - 1 ) * numel( options{ index_data }( index_target ).indices_elements_rx ) + options{ index_data }( index_target ).index_selected_rx_ref;
            e_B_tilde_ref{ index_data }{ index_target } = processing.signal( e_B_tilde{ index_data }{ index_target }.axis, e_B_tilde{ index_data }{ index_target }.samples( :, index_ref ) );

            %--------------------------------------------------------------
            % f) estimate calibration factors
            %--------------------------------------------------------------
            % reshape samples of PE responses
            e_B_samples = reshape( e_B_samples, [ abs( u_SA_window.axis ), numel( options{ index_data }( index_target ).indices_elements_rx ), numel( options{ index_data }( index_target ).indices_elements_tx ) ] );

            % ratios of transmitter electromechanical transfer functions
            filter = e_B_samples( :, :, options{ index_data }( index_target ).index_selected_tx_ref );
            filter_abs_squared = abs( filter ).^2;
            filter_abs_squared_max = max( filter_abs_squared );
            cal_tx_samples = squeeze( mean( e_B_samples .* conj( filter ) ./ ( filter_abs_squared + options{ index_data }( index_target ).epsilon_squared_cal * filter_abs_squared_max ), 2 ) );

            % Fourier synthesis
            cal_tx = processing.signal_matrix( e_B.axis, cal_tx_samples );
            cal_tx_tilde{ index_data }{ index_target } = inverse_fourier_transform( cal_tx, - round( 0.5 * abs( interval_t_tof_quantized ) / u_SA_tilde{ index_data }( 1 ).axis.delta ), u_SA_tilde{ index_data }( 1 ).axis.delta );

            % apply windows (use original time window)
            cal_tx_tilde{ index_data }{ index_target } = cut_out( cal_tx_tilde{ index_data }{ index_target }, cal_tx_tilde{ index_data }{ index_target }.axis.members( 1 ), cal_tx_tilde{ index_data }{ index_target }.axis.members( end ), [], options{ index_data }( index_target ).setting_window );

            % ratios of receiver electromechanical transfer functions
            filter = e_B_samples( :, options{ index_data }( index_target ).index_selected_rx_ref, : );
            filter_abs_squared = abs( filter ).^2;
            filter_abs_squared_max = max( filter_abs_squared );
            cal_rx_samples = mean( e_B_samples .* conj( filter ) ./ ( filter_abs_squared + options{ index_data }( index_target ).epsilon_squared_cal * filter_abs_squared_max ), 3 );

            % Fourier synthesis
            cal_rx = processing.signal_matrix( e_B.axis, cal_rx_samples );
            cal_rx_tilde{ index_data }{ index_target } = inverse_fourier_transform( cal_rx, - round( 0.5 * abs( interval_t_tof_quantized ) / u_SA_tilde{ index_data }( 1 ).axis.delta ), u_SA_tilde{ index_data }( 1 ).axis.delta );

            % apply windows (use original time window)
            cal_rx_tilde{ index_data }{ index_target } = cut_out( cal_rx_tilde{ index_data }{ index_target }, cal_rx_tilde{ index_data }{ index_target }.axis.members( 1 ), cal_rx_tilde{ index_data }{ index_target }.axis.members( end ), [], options{ index_data }( index_target ).setting_window );

            % prediction errors
            e_B_samples_hat = e_B_samples( :, options{ index_data }( index_target ).index_selected_rx_ref, options{ index_data }( index_target ).index_selected_tx_ref ) .* reshape( cal_tx_samples, [ abs( u_SA_window.axis ), 1, numel( options{ index_data }( index_target ).indices_elements_tx ) ] ) .* cal_rx_samples;
            e_B_samples_error = e_B_samples_hat - e_B_samples;

            rel_RMSE_local{ index_data }{ index_target } = squeeze( vecnorm( e_B_samples_error, 2, 1 ) ./ vecnorm( e_B_samples, 2, 1 ) );
            rel_RMSE_local_mean = mean( rel_RMSE_local{ index_data }{ index_target }( : ) );
            rel_RMSE_local_std_dev = std( rel_RMSE_local{ index_data }{ index_target }( : ) );
            rel_RMSE_local_min = min( rel_RMSE_local{ index_data }{ index_target }( : ) );
            rel_RMSE_local_max = max( rel_RMSE_local{ index_data }{ index_target }( : ) );

            rel_RMSE_global = norm( e_B_samples_error( : ) ) / norm( e_B_samples( : ) );

            %--------------------------------------------------------------
            % g) illustrate results
            %--------------------------------------------------------------
            figure( index_data );
            subplot( 5, N_targets, index_target );
            plot( double( e_B_tilde_ref{ index_data }{ index_target }.axis.members ), double( e_B_tilde_ref{ index_data }{ index_target }.samples ), '-b' );
            title( 'Reference' );
            subplot( 5, N_targets, N_targets + index_target );
            imagesc( options{ index_data }( index_target ).indices_elements_tx, double( cal_tx.axis.members ), abs( cal_tx.samples ) );
            title( 'TX calibration' );
            subplot( 5, N_targets, 2 * N_targets + index_target );
            imagesc( options{ index_data }( index_target ).indices_elements_rx, double( cal_rx.axis.members ), abs( cal_rx.samples ) );
            title( 'RX calibration' );
            subplot( 5, N_targets, 3 * N_targets + index_target );
            imagesc( rel_RMSE_local{ index_data }{ index_target } );
            title( {'Rel. RMSE', sprintf( 'mean: %.2f +- %.2f %%', rel_RMSE_local_mean * 1e2, rel_RMSE_local_std_dev * 1e2 ), sprintf( 'min/max: %.2f / %.2f %%', rel_RMSE_local_min * 1e2, rel_RMSE_local_max * 1e2 ), sprintf( 'global: %.2f %%', rel_RMSE_global * 1e2 ) } );
            subplot( 5, N_targets, 4 * N_targets + index_target );
            plot( double( e_B_tilde_mean{ index_data }{ index_target }.axis.members ), double( e_B_tilde_mean{ index_data }{ index_target }.samples ), '-b', ...
                  double( e_B_tilde_mean{ index_data }{ index_target }.axis.members ), double( e_B_tilde_mean{ index_data }{ index_target }.samples - e_B_tilde_std_dev{ index_data }{ index_target }.samples ), '--r', ...
                  double( e_B_tilde_mean{ index_data }{ index_target }.axis.members ), double( e_B_tilde_mean{ index_data }{ index_target }.samples + e_B_tilde_std_dev{ index_data }{ index_target }.samples ), '--r' );
            title( 'Mean +- Std. Dev.' );

        end % for index_target = 1:N_targets

        % concatenate signal matrices
        e_B_tilde_ref{ index_data } = reshape( [ e_B_tilde_ref{ index_data }{ : } ], size( states{ index_data } ) );
        cal_tx_tilde{ index_data } = reshape( [ cal_tx_tilde{ index_data }{ : } ], size( states{ index_data } ) );
        cal_rx_tilde{ index_data } = reshape( [ cal_rx_tilde{ index_data }{ : } ], size( states{ index_data } ) );
        e_B_tilde{ index_data } = reshape( [ e_B_tilde{ index_data }{ : } ], size( states{ index_data } ) );
        e_B_tilde_mean{ index_data } = reshape( [ e_B_tilde_mean{ index_data }{ : } ], size( states{ index_data } ) );
        e_B_tilde_std_dev{ index_data } = reshape( [ e_B_tilde_std_dev{ index_data }{ : } ], size( states{ index_data } ) );

    end % for index_data = 1:numel( u_SA_tilde )

	% avoid cell arrays for single u_SA_tilde
	if isscalar( u_SA_tilde )
        e_B_tilde_ref = e_B_tilde_ref{ 1 };
        cal_tx_tilde = cal_tx_tilde{ 1 };
        cal_rx_tilde = cal_rx_tilde{ 1 };
        rel_RMSE_local = rel_RMSE_local{ 1 };
        e_B_tilde = e_B_tilde{ 1 };
        e_B_tilde_mean = e_B_tilde_mean{ 1 };
        e_B_tilde_std_dev = e_B_tilde_std_dev{ 1 };
	end

end % function [ e_B_tilde_ref, cal_tx_tilde, cal_rx_tilde, rel_RMSE_local, e_B_tilde, e_B_tilde_mean, e_B_tilde_std_dev ] = estimate_per_point_qsw( u_SA_tilde, xdc_array, states, options )
