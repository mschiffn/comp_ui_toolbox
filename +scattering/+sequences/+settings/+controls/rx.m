%
% superclass for all transducer control settings in recording mode
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2019-11-28
%
classdef rx < scattering.sequences.settings.controls.common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        interval_t ( 1, 1 ) math.interval	% recording time interval
        interval_f ( 1, 1 ) math.interval	% frequency interval

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = rx( indices_active, impulse_responses, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if nargin == 0
                indices_active = 1;
                impulse_responses = physical_values.impulse_response( discretizations.set_discrete_time_regular( 0, 0, physical_values.time(1) ), physical_values.physical_value(1) );
                intervals_t = math.interval( physical_values.time( 0 ), physical_values.time( 1 ) );
                intervals_f = math.interval( physical_values.frequency( 1 ), physical_values.frequency( 2 ) );
            end

            % ensure cell array for indices_active
            if ~iscell( indices_active )
                indices_active = { indices_active };
            end

            % superclass ensures finite positive integers for indices_active

            % superclass ensures class discretizations.signal_matrix for impulse_responses

            % ensure class math.interval for intervals_t
            if ~( isa( intervals_t, 'math.interval' ) && isa( intervals_f, 'math.interval' ) )
                errorStruct.message = 'intervals_t and intervals_f must be math.interval!';
                errorStruct.identifier = 'rx:NoIntervals';
                error( errorStruct );
            end

            % multiple indices_active / single intervals_t
            if ~isscalar( indices_active ) && isscalar( intervals_t )
                intervals_t = repmat( intervals_t, size( indices_active ) );
            end

            % multiple indices_active / single intervals_f
            if ~isscalar( indices_active ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( indices_active ) );
            end

% TODO: determine frequency intervals / assertion: f_lb > 0, f_ub >= f_lb + 1 / T_rec
%             [ intervals_t, hulls ] = determine_interval_t( object );

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( indices_active, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) create transducer control settings in recording mode
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.settings.controls.common( indices_active, impulse_responses );

            % iterate transducer control settings in recording mode
            for index_object = 1:numel( objects )

                % ensure time interval
                if ~isa( intervals_t( index_object ).lb, 'physical_values.time' )
                    errorStruct.message = sprintf( 'Bounds of intervals_t( %d ) must be physical_values.time!', index_object );
                    errorStruct.identifier = 'rx:NoTimeInterval';
                    error( errorStruct );
                end

                % ensure frequency interval
                if ~isa( intervals_f( index_object ).lb, 'physical_values.frequency' )
                    errorStruct.message = sprintf( 'Bounds of intervals_f( %d ) must be physical_values.frequency!', index_object );
                    errorStruct.identifier = 'rx:NoFrequencyInterval';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).interval_t = intervals_t( index_object );
                objects( index_object ).interval_f = intervals_f( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = rx( indices_active, impulse_responses, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % spectral discretization (overload discretize method)
        %------------------------------------------------------------------
        function settings_rx = discretize( settings_rx, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure correct number of arguments
            if nargin ~= 1 && nargin ~= 3
                errorStruct.message     = 'Either one or three arguments are required!';
                errorStruct.identifier	= 'discretize:Arguments';
                error( errorStruct );
            end

            % specify recording time and frequency intervals
            if nargin == 1

                % use intervals from transducer control settings
                Ts_ref = reshape( abs( [ settings_rx.interval_t ] ), size( settings_rx ) );
                intervals_f = reshape( [ settings_rx.interval_f ], size( settings_rx ) );

            else

                % use external intervals
                Ts_ref = varargin{ 1 };
                intervals_f = varargin{ 2 };

            end % if nargin == 1

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples via superclass method
            %--------------------------------------------------------------
            settings_rx = discretize@scattering.sequences.settings.controls.common( settings_rx, Ts_ref, intervals_f );

        end % function settings_rx = discretize( settings_rx, varargin )

        %------------------------------------------------------------------
        % convex hulls of all intervals
        %------------------------------------------------------------------
        function [ interval_hull_t, interval_hull_f ] = hulls( settings_rx )
% TODO: quantize intervals? check T_s foc compatibility
            % convex hull of all recording time intervals
            interval_hull_t = hull( [ settings_rx.interval_t ] );

            % convex hull of all frequency intervals
            interval_hull_f = hull( [ settings_rx.interval_f ] );

        end % function [ interval_hull_t, interval_hull_f ] = hulls( settings_rx )

        %------------------------------------------------------------------
        % estimate recording time intervals
        %------------------------------------------------------------------
        function intervals_t = determine_interval_t( settings_rx, setup, settings_tx )

            %--------------------------------------------------------------
            % 1.) lower and upper bounds on the times-of-flight
            %--------------------------------------------------------------
            intervals_tof = times_of_flight( setup, { settings_tx.indices_active }, { settings_rx.indices_active } );

            %--------------------------------------------------------------
            % 2.) estimate support of each mix
            %--------------------------------------------------------------
            N_incident = numel( object.settings );
            intervals_t = cell( N_incident, 1 );
            hulls = repmat( tof( 1, 1 ), [ N_incident, 1 ] );

            for index_incident = 1:N_incident

                % indices of active tx elements
                indices_tx_act = object.settings( index_incident ).tx.indices_active;
                N_elements_tx = numel( indices_tx_act );

                % determine support of each mix
                N_mix = numel( object.settings( index_incident ).mixes );

                % initialize lower and upper bounds on the support
                t_lbs = physical_values.time( zeros( 1, N_mix ) );
                t_ubs = physical_values.time( zeros( 1, N_mix ) );

                for index_mix = 1:N_mix

                    % indices of active rx elements
                    indices_rx_act = object.settings( index_incident ).rx( index_mix ).indices_active;
                    N_elements_rx = numel( indices_rx_act );

                    % allocate memory
                    t_lbs_all = physical_values.time( zeros( N_elements_tx, N_elements_rx ) );
                    t_ubs_all = physical_values.time( zeros( N_elements_tx, N_elements_rx ) );

                    % check all combinations of active tx and rx elements
                    for index_tx = 1:N_elements_tx

                        % index of tx array element
                        index_element_tx = indices_tx_act( index_tx );

                        % support of excitation voltage
                        t_lb_tx_act = object.settings( index_incident ).tx.excitation_voltages( index_tx ).set_t.S( 1 ) + object.settings( index_incident ).tx.time_delays( index_tx );
                        t_ub_tx_act = object.settings( index_incident ).tx.excitation_voltages( index_tx ).set_t.S( end ) + object.settings( index_incident ).tx.time_delays( index_tx );

                        for index_rx = 1:N_elements_rx

                            % index of rx array element
                            index_element_rx = indices_rx_act( index_rx );

                            % support of impulse response
                            t_lb_rx_act = object.settings( index_incident ).rx( index_mix ).impulse_responses( index_rx ).set_t.S( 1 );
                            t_ub_rx_act = object.settings( index_incident ).rx( index_mix ).impulse_responses( index_rx ).set_t.S( end );

                            t_lbs_all( index_tx, index_rx ) = t_lb_tx_act + tof( index_element_tx, index_element_rx ).bounds( 1 ) + t_lb_rx_act;
                            t_ubs_all( index_tx, index_rx ) = t_ub_tx_act + tof( index_element_tx, index_element_rx ).bounds( 2 ) + t_ub_rx_act;

                        end % for index_rx = 1:N_elements_rx
                    end % for index_tx = 1:N_elements_tx

                    t_lbs( index_mix ) = min( t_lbs_all );
                    t_ubs( index_mix ) = max( t_ubs_all );

                end % for index_mix = 1:N_mix

                % create time intervals for all mixes
                intervals_t{ index_incident } = math.interval_time( t_lbs, t_ubs );

            end % for index_incident = 1:N_incident

        end % function [ intervals_t, hulls ] = determine_interval_t( object )

        %------------------------------------------------------------------
        % compute numbers of observations
        %------------------------------------------------------------------
        function N_observations = compute_N_observations( settings_rx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.settings.controls.rx
            if ~isa( settings_rx, 'scattering.sequences.settings.controls.rx' )
                errorStruct.message = 'settings_rx must be scattering.sequences.settings.controls.rx!';
                errorStruct.identifier = 'compute_N_observations:NoSettingsRx';
                error( errorStruct );
            end

            % extract impulse responses
            impulse_responses = { settings_rx.impulse_responses };

            % ensure class discretizations.signal_matrix
            N_signal_matrices = cellfun( @numel, impulse_responses );
            if any( N_signal_matrices ~= 1 )
                errorStruct.message = 'excitation_voltages and impulse_responses must be discretizations.signal_matrix!';
                errorStruct.identifier = 'compute_normal_velocities:NoSignalMatrices';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) extract numbers of observations
            %--------------------------------------------------------------
            % specify cell array for N_observations
            impulse_responses = reshape( [ settings_rx.impulse_responses ], size( settings_rx ) );
            axes = reshape( [ impulse_responses.axis ], size( settings_rx ) );
            N_observations = abs( axes );

        end % function N_observations = compute_N_observations( settings_rx )

	end % methods

end % classdef rx < scattering.sequences.settings.controls.common
