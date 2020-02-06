%
% superclass for all calibration options
%
% author: Martin F. Schiffner
% date: 2019-11-26
% modified: 2020-02-03
%
classdef (Abstract) common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        duration_window_t ( 1, 1 ) physical_values.time { mustBeNonempty } = physical_values.second( 2.5e-6 )	% time duration of the correlation window
        setting_window ( 1, 1 ) auxiliary.setting_window { mustBeNonempty } = auxiliary.setting_window          % window settings
        display ( 1, 1 ) logical { mustBeNonempty } = 1;                                                        % display results of estimate

        % dependent properties
        interval_window_t ( 1, 1 ) math.interval = math.interval( physical_values.second( 0 ), physical_values.second( 2.5e-6 ) ); % correlation window

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = common( durations_window_t, settings_window )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no argument
            if nargin == 0
                return;
            end

            % property validation function ensures class physical_values.time for durations_window_t

            % ensure nonempty settings_window
            if nargin < 2 || isempty( settings_window )
                settings_window = repmat( auxiliary.setting_window, size( durations_window_t ) );
            end

            % property validation function ensures class auxiliary.setting_window for settings_window

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( durations_window_t, settings_window );

            %--------------------------------------------------------------
            % 2.) create calibration options
            %--------------------------------------------------------------
            % repeat default calibration options
            objects = repmat( objects, size( durations_window_t ) );

            % iterate calibration options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).duration_window_t = durations_window_t( index_object );
                objects( index_object ).setting_window = settings_window( index_object );

                % set dependent properties
                objects( index_object ).interval_window_t = math.interval( 0 * objects( index_object ).duration_window_t, objects( index_object ).duration_window_t );

            end % for index_object = 1:numel( objects )

        end % function objects = common( durations_window_t, settings_window )

	end % methods

end % classdef (Abstract) common
