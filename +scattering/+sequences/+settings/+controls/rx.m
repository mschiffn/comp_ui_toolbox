%
% superclass for all system controls in receive mode (mixer settings)
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2020-04-07
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
            % ensure correct number of arguments
            narginchk( 2, 4 );

            % ensure cell array for indices_active
            if ~iscell( indices_active )
                indices_active = { indices_active };
            end

            % superclass ensures finite positive integers for indices_active

            % superclass ensures class processing.signal_matrix for impulse_responses

            % ensure nonempty intervals_t
            if nargin < 3 || isempty( intervals_t )
                intervals_t = math.interval( physical_values.second( -Inf ), physical_values.second( Inf ) );
            end

            % ensure class math.interval for intervals_t
            if ~isa( intervals_t, 'math.interval' )
                errorStruct.message = 'intervals_t must be math.interval!';
                errorStruct.identifier = 'rx:NoTimeIntervals';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.time
            auxiliary.mustBeEqualSubclasses( 'physical_values.time', intervals_t.lb );

            % ensure nonempty intervals_f
            if nargin < 4 || isempty( intervals_f )
                intervals_f = math.interval( physical_values.hertz( 0 ), physical_values.hertz( Inf ) );
            end

            % ensure class math.interval for intervals_f
            if ~isa( intervals_f, 'math.interval' )
                errorStruct.message = 'intervals_f must be math.interval!';
                errorStruct.identifier = 'rx:NoFrequencyIntervals';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.frequency
            auxiliary.mustBeEqualSubclasses( 'physical_values.frequency', intervals_f.lb );

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
            % 2.) create mixer settings
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.settings.controls.common( indices_active, impulse_responses );

            % iterate transducer control settings in recording mode
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).interval_t = intervals_t( index_object );
                objects( index_object ).interval_f = intervals_f( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = rx( indices_active, impulse_responses, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % spectral discretization (overload discretize method)
        %------------------------------------------------------------------
        function settings_rx = discretize( settings_rx, Ts_ref, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure correct number of arguments
            if nargin ~= 1 && nargin ~= 3
                errorStruct.message = 'Either one or three arguments are required!';
                errorStruct.identifier = 'discretize:Arguments';
                error( errorStruct );
            end

            % specify recording time and frequency intervals
            if nargin == 1

                % use intervals from transducer control settings
                Ts_ref = reshape( abs( [ settings_rx.interval_t ] ), size( settings_rx ) );
                intervals_f = reshape( [ settings_rx.interval_f ], size( settings_rx ) );

            end % if nargin == 1

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples via superclass method
            %--------------------------------------------------------------
            settings_rx = discretize@scattering.sequences.settings.controls.common( settings_rx, Ts_ref, intervals_f );

        end % function settings_rx = discretize( settings_rx, Ts_ref, intervals_f )

        %------------------------------------------------------------------
        % intersect recording time intervals
        %------------------------------------------------------------------
        function settings_rx = intersect( settings_rx, intervals_t )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure correct number of input arguments
            narginchk( 2, 2 );

            % ensure class scattering.sequences.settings.controls.rx
            if ~isa( settings_rx, 'scattering.sequences.settings.controls.rx' )
                errorStruct.message = 'settings_rx must be scattering.sequences.settings.controls.rx!';
                errorStruct.identifier = 'intersect:NoSettingsRx';
                error( errorStruct );
            end

            % ensure class math.interval
            if ~isa( intervals_t, 'math.interval' )
                errorStruct.message = 'intervals_t must be math.interval!';
                errorStruct.identifier = 'intersect:NoTimeIntervals';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.time
            auxiliary.mustBeEqualSubclasses( 'physical_values.time', intervals_t.lb );

            % ensure equal number of dimensions and sizes
            [ settings_rx, intervals_t ] = auxiliary.ensureEqualSize( settings_rx, intervals_t );

            %--------------------------------------------------------------
            % 2.) intersect recording time intervals
            %--------------------------------------------------------------
            for index_object = 1:numel( settings_rx )

                % determine intersection
                t_lb = max( settings_rx( index_object ).interval_t.lb, intervals_t( index_object ).lb );
                t_ub = min( settings_rx( index_object ).interval_t.ub, intervals_t( index_object ).ub );

                % correct recording time interval
                settings_rx( index_object ).interval_t = math.interval( t_lb, t_ub );

            end % for index_object = 1:numel( settings_rx )

        end % function settings_rx = intersect( settings_rx, intervals_t )

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

            % ensure class processing.signal_matrix
            N_signal_matrices = cellfun( @numel, impulse_responses );
            if any( N_signal_matrices ~= 1 )
                errorStruct.message = 'excitation_voltages and impulse_responses must be processing.signal_matrix!';
                errorStruct.identifier = 'compute_N_observations:NoSignalMatrices';
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
