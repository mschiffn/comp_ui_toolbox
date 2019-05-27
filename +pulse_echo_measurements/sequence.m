%
% superclass for all sequential pulse-echo measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-05-25
%
classdef sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % independent properties
        setup %( 1, 1 ) pulse_echo_measurements.setup        % pulse-echo measurement setup
        settings %( :, : ) pulse_echo_measurements.setting	% pulse-echo measurement settings

        % dependent properties
        interval_hull_t ( 1, 1 ) math.interval	% hull of all recording time intervals
        interval_hull_f ( 1, 1 ) math.interval	% hull of all frequency intervals

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods ( Access = public )

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence( setups, settings )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setups, 'pulse_echo_measurements.setup' )
                errorStruct.message     = 'setups must be pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'sequence:NoSetup';
                error( errorStruct );
            end

            % ensure cell array for settings
            if ~iscell( settings )
                settings = { settings };
            end

            % multiple setups / single settings
            if ~isscalar( setups ) && isscalar( settings )
                settings = repmat( settings, size( setups ) );
            end

            % single setups / multiple settings
            if isscalar( setups ) && ~isscalar( settings )
                setups = repmat( setups, size( settings ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, settings );

            %--------------------------------------------------------------
            % 2.) create sequences of pulse-echo measurements
            %--------------------------------------------------------------
            % repeat default sequence
            objects = repmat( objects, size( setups ) );

            % iterate sequences
            for index_object = 1:numel( objects )

                % ensure class pulse_echo_measurements.setting
                if ~isa( settings{ index_object }, 'pulse_echo_measurements.setting' )
                    errorStruct.message     = 'settings must be pulse_echo_measurements.setting!';
                    errorStruct.identifier	= 'sequence:NoSetting';
                    error( errorStruct );
                end

                % TODO: ensure that settings are compatible w/ setup

                % set independent properties
                objects( index_object ).setup = setups( index_object );
                objects( index_object ).settings = settings{ index_object };

                % set dependent properties
                [ objects( index_object ).interval_hull_t, objects( index_object ).interval_hull_f ] = hulls( objects( index_object ).settings );

            end % for index_object = 1:numel( objects )

        end % function objects = sequence( setups, settings )

        %------------------------------------------------------------------
        % spatiospectral discretizations
        %------------------------------------------------------------------
        function spatiospectrals = discretize( sequences, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.options
            if ~isa( options, 'discretizations.options' )
                errorStruct.message = 'options must be discretizations.options!';
                errorStruct.identifier = 'discretize:NoOptions';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, options );

            %--------------------------------------------------------------
            % 2.) spatial discretizations
            %--------------------------------------------------------------
% TODO: conserve shape
            setups = reshape( [ sequences.setup ], size( sequences ) );
            spatials = discretize( setups, [ options.spatial ] );

            %--------------------------------------------------------------
            % 3.) spectral discretizations
            %--------------------------------------------------------------
            % specify cell array for spectrals
            spectrals = cell( size( sequences ) );

            % iterate sequential pulse-echo measurements
            for index_object = 1:numel( sequences )

                % discretize pulse-echo measurement settings
                spectrals{ index_object } = discretize( sequences( index_object ).settings, options( index_object ).spectral );

            end % for index_object = 1:numel( sequences )

            %--------------------------------------------------------------
            % 4.) create spatiospectral discretizations
            %--------------------------------------------------------------
            spatiospectrals = discretizations.spatiospectral( spatials, spectrals );

        end % function spatiospectrals = discretize( sequences, options )

        %------------------------------------------------------------------
        % apply windows to mixed voltage signals
        %------------------------------------------------------------------
        function u_M = apply_window( sequence, u_M, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for u_M
            if ~iscell( u_M )
                u_M = { u_M };
            end

            % ensure nonempty setting_window
            if nargin >= 3 && isa( varargin{ 1 }, 'auxiliary.setting_window' )
                setting_window = varargin{ 1 };
            else
                setting_window = auxiliary.setting_window;
            end

            %--------------------------------------------------------------
            % 2.) apply window functions
            %--------------------------------------------------------------
            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( sequence.settings )

                % extract recording time intervals for all mixed voltage signals
                intervals_t = reshape( [ sequence.settings( index_measurement ).rx.interval_t ], size( sequence.settings( index_measurement ).rx ) );

                % extract lower and upper bounds on the recording time intervals
                lbs = reshape( [ intervals_t.lb ], size( intervals_t ) );
                ubs = reshape( [ intervals_t.ub ], size( intervals_t ) );

                % check data structure
                if isa( u_M{ index_measurement }, 'discretizations.signal' )

                    %------------------------------------------------------
                    % a) single signals
                    %------------------------------------------------------
                    % extract axes of all mixed voltage signals
                    axes = reshape( [ u_M{ index_measurement }.axis ], size( u_M{ index_measurement } ) );
                    samples_win = cell( size( u_M{ index_measurement } ) );

                    % iterate mixed voltage signals
                    for index_mix = 1:numel( sequence.settings( index_measurement ).rx )

                        % determine lower and upper bounds on the windows
                        lbs_max = max( lbs( index_mix ), u_M{ index_measurement }( index_mix ).axis.members( 1 ) );
                        ubs_min = min( ubs( index_mix ), u_M{ index_measurement }( index_mix ).axis.members( end ) );

                        % number of samples in the windows
                        indicator_lb = u_M{ index_measurement }( index_mix ).axis.members >= lbs_max;
                        indicator_ub = u_M{ index_measurement }( index_mix ).axis.members <= ubs_min;
                        indicator = indicator_lb & indicator_ub;
                        N_samples_window = sum( indicator, 2 );

                        % generate and apply window functions
                        test = window( setting_window.handle, N_samples_window, setting_window.parameters{ : } )';
                        samples_win = u_M{ index_measurement }( index_mix ).samples;
                        samples_win( ~indicator ) = 0;
                        samples_win( indicator ) = samples_win( indicator ) .* test;

                    end % for index_mix = 1:numel( sequence.settings( index_measurement ).rx )

                    % create signals
                    u_M{ index_measurement } = discretizations.signal( axes, samples_win );

                else

                    %------------------------------------------------------
                    % b) signal matrix
                    %------------------------------------------------------
                    % determine lower and upper bounds on all windows
                    lbs_max = max( lbs, u_M{ index_measurement }.axis.members( 1 ) );
                    ubs_min = min( ubs, u_M{ index_measurement }.axis.members( end ) );

                    % determine lower and upper bounds on axis
                    lbs_max_min = min( lbs_max, [], 'all' );
                    ubs_min_max = max( ubs_min, [], 'all' );

                    % cut out from signal matrix
                    u_M{ index_measurement } = cut_out( u_M{ index_measurement }, lbs_max_min, ubs_min_max );

                    % numbers of samples in all windows
                    indicator_lb = repmat( u_M{ index_measurement }.axis.members, [ numel( intervals_t ), 1 ] ) >= lbs_max( : );
                    indicator_ub = repmat( u_M{ index_measurement }.axis.members, [ numel( intervals_t ), 1 ] ) <= ubs_min( : );
                    indicator = indicator_lb & indicator_ub;
                    N_samples_window = sum( indicator, 2 );

                    % generate and apply window functions
                    samples = u_M{ index_measurement }.samples;
                    samples( ~indicator ) = 0;

                    % iterate mixed voltage signals
                    for index_mix = 1:numel( sequence.settings( index_measurement ).rx )

                        % window function gateway
                        samples_window = window( setting_window.handle, N_samples_window( index_mix ), setting_window.parameters{ : } )';

                        % apply window function
                        samples( index_mix, indicator( index_mix, : ) ) = samples( index_mix, indicator( index_mix, : ) ) .* samples_window;

                    end % for index_mix = 1:numel( sequence.settings( index_measurement ).rx )

                    % periodicity renders last sample redundant
                    axis = remove_last( u_M{ index_measurement }.axis );
                    samples = samples( :, 1:( end - 1 ) );

                    % create signal matrix
                    u_M{ index_measurement } = discretizations.signal_matrix( axis, samples );

                end % if isa( u_M{ index_measurement }, 'discretizations.signal' )

            end % for index_measurement = 1:numel( sequence.settings )

            % avoid cell array for single pulse-echo measurement
            if isscalar( u_M )
                u_M = u_M{ 1 };
            end

        end % function u_M = apply_window( sequence, u_M, varargin )

	end % methods

end % classdef sequence
