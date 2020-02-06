function [ e_B_tilde, e_B_tilde_mean, e_B_tilde_std_dev ] = estimate_per_point_qpw( u_rx_tilde_qpw, xdc_array, states, options )
%
% estimate the pulse-echo responses for multiple point-like targets
% (cf. [1])
%
% [1] R. Huang and Lester W. Schmerr Jr., "Characterization of the system functions of ultrasonic linear phased array inspection systems",
% Ultrasonics, Vol. 49, 2009, pp. 219-225 (doi:10.1016/j.ultras.2008.08.004)
%
% author: Martin F. Schiffner
% date: 2019-06-13
% modified: 2020-02-06

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure class processing.signal_matrix
    if ~isa( u_rx_tilde_qpw, 'processing.signal_matrix' )
        errorStruct.message = 'u_rx_tilde_qpw must be processing.signal_matrix!';
        errorStruct.identifier = 'estimate_per_point_qpw:NoSignalMatrices';
        error( errorStruct );
    end

	% ensure class scattering.sequences.setups.transducers.array_planar_regular
	if ~isa( xdc_array, 'scattering.sequences.setups.transducers.array_planar_regular' )
        errorStruct.message = 'xdc_array must be scattering.sequences.setups.transducers.array_planar_regular!';
        errorStruct.identifier = 'estimate_per_point_qpw:NoRegularPlanarArray';
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
	% 2.) check arguments
	%----------------------------------------------------------------------
	% specify cell arrays
    e_B_tilde = cell( size( u_rx_tilde_qpw ) );
    e_B_tilde_mean = cell( size( u_rx_tilde_qpw ) );
    e_B_tilde_std_dev = cell( size( u_rx_tilde_qpw ) );

	% iterate signal matrices
	for index_data = 1:numel( u_rx_tilde_qpw )

        %------------------------------------------------------------------
        % a) check arguments
        %------------------------------------------------------------------
        % ensure valid number of signals
        if u_rx_tilde_qpw( index_data ).N_signals ~= xdc_array( index_data ).N_elements
            errorStruct.message = sprintf( 'The number of signals in u_rx_tilde_qpw( %d ) must equal the number of elements in xdc_array( %d )!', index_data, index_data );
            errorStruct.identifier = 'estimate_per_point_qpw:InvalidNumberOfSignals';
            error( errorStruct );
        end

        % ensure class calibration.state
        if ~isa( states{ index_data }, 'calibration.state' )
            errorStruct.message = 'states{ index_data } must be calibration.state!';
            errorStruct.identifier = 'estimate_per_point_qpw:NoStates';
            error( errorStruct );
        end

        % ensure class calibration.options
        if ~isa( options{ index_data }, 'calibration.options' )
            errorStruct.message = 'options{ index_data } must be calibration.options!';
            errorStruct.identifier = 'estimate_per_point_qpw:NoOptions';
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
        times_of_flight = calibration.function_tof( xdc_array( index_data ).positions_ctr, states{ index_data } );

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
        e_B_tilde{ index_data } = cell( size( states{ index_data } ) );
        e_B_tilde_mean{ index_data } = cell( size( states{ index_data } ) );
        e_B_tilde_std_dev{ index_data } = cell( size( states{ index_data } ) );

        % iterate point-like targets
        for index_target = 1:N_targets

            %--------------------------------------------------------------
            % a) compute time intervals based on predicted times-of-flight and waveform center
            %--------------------------------------------------------------
            % predict waveform starts using times-of-flight
            tof_start = times_of_flight{ index_target }( options{ index_data }( index_target ).indices_elements );

            % create time intervals for all active elements
            intervals_t = move( options{ index_data }( index_target ).interval_window_t, tof_start + abs( options{ index_data }( index_target ).interval_window_t ) / 2 );

            %--------------------------------------------------------------
            % b) cut out waveforms
            %--------------------------------------------------------------
            % cut out waveforms (apply windows)
            u_rx_tilde_qpw_window = cut_out( u_rx_tilde_qpw( index_data ), [ intervals_t.lb ], [ intervals_t.ub ], num2cell( options{ index_data }( index_target ).indices_elements ), options{ index_data }( index_target ).setting_window );

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
            interval_t_tof_quantized = quantize( math.interval( t_tof_lbs_min, t_tof_ubs_max + options{ index_data }( index_target ).interval_window_t.ub ), u_rx_tilde_qpw( index_data ).axis.delta );
            u_rx_qpw_window = fourier_coefficients( u_rx_tilde_qpw_window, abs( interval_t_tof_quantized ), options{ index_data }( index_target ).interval_f );
            u_rx_qpw_window = merge( u_rx_qpw_window );

            % compute spatial transfer functions
            h_transfer = transfer_function( setup, u_rx_qpw_window.axis, ( 1:setup.xdc_array.N_elements ) );
            h_tx = sum( h_transfer );

            % compute prefactors
% TODO: check units!
            prefactors = compute_prefactors( setup, u_rx_qpw_window.axis );

            %--------------------------------------------------------------
            % e) estimate pulse-echo response
            %--------------------------------------------------------------
            epsilon_squared = 7.5e-3;
            e_B_samples = u_rx_qpw_window.samples;

            % iterate specified elements
            for index_selected = 1:numel( options{ index_data }( index_target ).indices_elements )

                % index of the array element
                index_element = options{ index_data }( index_target ).indices_elements( index_selected );

                % round-trip transfer function
                h_rx_tx = double( prefactors.samples .* h_tx.samples .* h_transfer( index_element ).samples );

                % deconvolve waveform
                h_rx_tx_abs_squared = abs( h_rx_tx ).^2;
                h_rx_tx_abs_squared_max = max( h_rx_tx_abs_squared );
                e_B_samples( :, index_selected ) = u_rx_qpw_window.samples( :, index_selected ) .* conj( h_rx_tx ) ./ ( h_rx_tx_abs_squared + epsilon_squared * h_rx_tx_abs_squared_max );

%                 figure( 1 );
%                 plot( double( u_rx_qpw_window.axis.members ), abs( u_rx_qpw_window.samples( :, index_selected ) ), double( u_rx_qpw_window.axis.members ), abs( conj( h_rx_tx ) ./ ( h_rx_tx_abs_squared + epsilon_squared * h_rx_tx_abs_squared_max ) ), double( u_rx_qpw_window.axis.members ), abs( e_B_samples( :, index_selected ) ) );

            end % for index_selected = 1:numel( options{ index_data }( index_target ).indices_elements )

            % Fourier synthesis
            e_B = processing.signal_matrix( u_rx_qpw_window.axis, e_B_samples );
            e_B_tilde{ index_data }{ index_target } = signal( e_B, 0, u_rx_tilde_qpw( index_data ).axis.delta );

            % apply windows (use original time window)
            e_B_tilde{ index_data }{ index_target } = cut_out( e_B_tilde{ index_data }{ index_target }, options{ index_data }( index_target ).interval_window_t.lb, options{ index_data }( index_target ).interval_window_t.ub, [], options{ index_data }( index_target ).setting_window );

            % compute mean and standard deviation
            e_B_tilde_mean{ index_data }{ index_target } = processing.signal( e_B_tilde{ index_data }{ index_target }.axis, mean( e_B_tilde{ index_data }{ index_target }.samples, 2 ) );
            e_B_tilde_std_dev{ index_data }{ index_target } = processing.signal( e_B_tilde{ index_data }{ index_target }.axis, sqrt( var( e_B_tilde{ index_data }{ index_target }.samples, [], 2 ) ) );

            %--------------------------------------------------------------
            % f) illustrate results
            %--------------------------------------------------------------
            figure( index_data );
            subplot( 3, N_targets, index_target );
            imagesc( options{ index_data }( index_target ).indices_elements, double( e_B_tilde{ index_data }{ index_target }.axis.members ), double( e_B_tilde{ index_data }{ index_target }.samples ) );
            title( 'P-E Responses' );
            subplot( 3, N_targets, N_targets + index_target );
            plot( double( e_B_tilde{ index_data }{ index_target }.axis.members ), double( e_B_tilde{ index_data }{ index_target }.samples ) );
            title( 'P-E Responses' );
            subplot( 3, N_targets, 2 * N_targets + index_target );
            plot( double( e_B_tilde_mean{ index_data }{ index_target }.axis.members ), double( e_B_tilde_mean{ index_data }{ index_target }.samples ), '-b', ...
                  double( e_B_tilde_mean{ index_data }{ index_target }.axis.members ), double( e_B_tilde_mean{ index_data }{ index_target }.samples - e_B_tilde_std_dev{ index_data }{ index_target }.samples ), '--r', ...
                  double( e_B_tilde_mean{ index_data }{ index_target }.axis.members ), double( e_B_tilde_mean{ index_data }{ index_target }.samples + e_B_tilde_std_dev{ index_data }{ index_target }.samples ), '--r' );
            title( 'Mean +- Std. Dev.' );

        end % for index_target = 1:N_targets

        % concatenate signal matrices
        e_B_tilde{ index_data } = reshape( [ e_B_tilde{ index_data }{ : } ], size( states{ index_data } ) );
        e_B_tilde_mean{ index_data } = reshape( [ e_B_tilde_mean{ index_data }{ : } ], size( states{ index_data } ) );
        e_B_tilde_std_dev{ index_data } = reshape( [ e_B_tilde_std_dev{ index_data }{ : } ], size( states{ index_data } ) );

    end % for index_data = 1:numel( u_rx_tilde_qpw )

	% avoid cell arrays for single u_rx_tilde_qpw
	if isscalar( u_rx_tilde_qpw )
        e_B_tilde = e_B_tilde{ 1 };
        e_B_tilde_mean = e_B_tilde_mean{ 1 };
        e_B_tilde_std_dev = e_B_tilde_std_dev{ 1 };
    end

end % function [ e_B_tilde, e_B_tilde_mean, e_B_tilde_std_dev ] = estimate_per_point_qpw( u_rx_tilde_qpw, xdc_array, states, options )
