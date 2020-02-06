%
% superclass for all calibration options for QSWs
%
% author: Martin F. Schiffner
% date: 2020-02-03
% modified: 2020-02-03
%
classdef (Abstract) common_qsw < calibration.options.common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        indices_elements_tx ( :, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = (1:128)         % indices of tx elements
        indices_elements_rx ( :, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = (1:128)         % indices of rx elements

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = common_qsw( durations_window_t, indices_elements_tx, indices_elements_rx, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class physical_values.time for durations_window_t

            % ensure cell array for indices_elements_tx
            if ~iscell( indices_elements_tx )
                indices_elements_tx = { indices_elements_tx };
            end

            % property validation function ensures nonempty positive integers for indices_elements_tx

            % ensure cell array for indices_elements_rx
            if ~iscell( indices_elements_rx )
                indices_elements_rx = { indices_elements_rx };
            end

            % property validation function ensures nonempty positive integers for indices_elements_rx

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( durations_window_t, indices_elements_tx, indices_elements_rx );

            %--------------------------------------------------------------
            % 2.) create calibration options for QSWs
            %--------------------------------------------------------------
            % constructor of superclass
            objects@calibration.options.common( durations_window_t, varargin{ : } );

            % iterate calibration options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).indices_elements_tx = indices_elements_tx{ index_object };
                objects( index_object ).indices_elements_rx = indices_elements_rx{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = common_qsw( durations_window_t, indices_elements_tx, indices_elements_rx, varargin )

	end % methods

end % classdef (Abstract) common_qsw < calibration.options.common
