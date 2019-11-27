%
% superclass for all calibration options
%
% author: Martin F. Schiffner
% date: 2019-11-26
% modified: 2019-11-26
%
classdef (Abstract) common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        duration_window_t ( 1, 1 ) physical_values.time { mustBeNonempty } = physical_values.second( 2.5e-6 )	% time duration of window
        indices_elements_tx ( 1, : ) double { mustBePositive, mustBeInteger, mustBeNonempty } = (1:128)         % indices of tx elements
        indices_elements_rx ( 1, : ) double { mustBePositive, mustBeInteger, mustBeNonempty } = (1:128)         % indices of rx elements
        setting_window ( 1, 1 ) auxiliary.setting_window { mustBeNonempty } = auxiliary.setting_window          % window settings

        % dependent properties
        interval_window_t ( 1, 1 ) math.interval = math.interval( physical_values.second( 0 ), physical_values.second( 2.5e-6 ) );

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = common( durations_window_t, indices_elements_tx, indices_elements_rx, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no argument
            if nargin == 0
                return;
            end

            % property validation function ensures class physical_values.time for durations_window_t

            % ensure cell array for indices_elements_tx
            if ~iscell( indices_elements_tx )
                indices_elements_tx = { indices_elements_tx };
            end

            % property validation function ensures positive integers for indices_elements_tx

            % ensure cell array for indices_elements_rx
            if ~iscell( indices_elements_rx )
                indices_elements_rx = { indices_elements_rx };
            end

            % property validation function ensures positive integers for indices_elements_rx

            % ensure nonempty settings_window
            if nargin >= 4 && ~isempty( varargin{ 1 } )
                settings_window = varargin{ 1 };
            else
                settings_window = repmat( auxiliary.setting_window, size( durations_window_t ) );
            end

            % property validation function ensures class auxiliary.setting_window for settings_window

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( durations_window_t, indices_elements_tx, indices_elements_rx, settings_window );

            %--------------------------------------------------------------
            % 2.) create calibration options
            %--------------------------------------------------------------
            % repeat default calibration options
            objects = repmat( objects, size( durations_window_t ) );

            % iterate calibration options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).duration_window_t = durations_window_t( index_object );
                objects( index_object ).indices_elements_tx = indices_elements_tx{ index_object };
                objects( index_object ).indices_elements_rx = indices_elements_rx{ index_object };
                objects( index_object ).setting_window = settings_window( index_object );

                % set dependent properties
                objects( index_object ).interval_window_t = math.interval( 0 * objects( index_object ).duration_window_t, objects( index_object ).duration_window_t );

            end % for index_object = 1:numel( objects )

        end % function objects = common( durations_window_t, indices_elements_tx, indices_elements_rx, varargin )

	end % methods

end % classdef (Abstract) common
