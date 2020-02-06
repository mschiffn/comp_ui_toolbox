%
% superclass for all sound speed estimation options (QSW method)
%
% author: Martin F. Schiffner
% date: 2019-11-26
% modified: 2019-01-23
%
classdef sos_qsw < calibration.options.common_qsw

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        time_shift_ctr ( 1, 1 ) physical_values.time = physical_values.second( 19 / 40e6 )      % time shift to waveform center
        factor_interp ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 30    % interpolation factor

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sos_qsw( durations_window_t, time_shifts_ctr, indices_elements_tx, indices_element_lb, indices_element_ub, factors_interp, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class physical_values.time for durations_window_t

            % property validation function ensures class physical_values.time for time_shifts_ctr

            % superclass ensures cell array of positive integers for indices_elements_tx

            % ensure positive integers for indices_element_lb
            mustBeInteger( indices_element_lb );
            mustBePositive( indices_element_lb );

            % ensure positive integers for indices_element_ub
            mustBeInteger( indices_element_ub );
            mustBePositive( indices_element_ub );

            % ensure strictly increasing bounds
            if any( indices_element_lb( : ) >= indices_element_ub( : ) )
                errorStruct.message = 'indices_element_ub must exceed indices_element_lb!';
                errorStruct.identifier = 'sos_qsw:LowerBoundsExceedUpperBounds';
                error( errorStruct );
            end

            % ensure nonempty factors_interp
            if nargin < 6 || isempty( factors_interp )
                factors_interp = repmat( 30, size( durations_window_t ) );
            end

            % property validation function ensures nonempty positive integers for factors_interp

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( durations_window_t, time_shifts_ctr, indices_element_lb, indices_element_ub, factors_interp );

            %--------------------------------------------------------------
            % 2.) create sound speed estimation options
            %--------------------------------------------------------------
            % define indices of rx elements (cross-correlation method requires contiguous indices)
            indices_elements_rx = cell( size( durations_window_t ) );
            for index_object = 1:numel( durations_window_t )
                indices_elements_rx{ index_object } = ( indices_element_lb( index_object ):indices_element_ub( index_object ) );
            end

            % constructor of superclass
            objects@calibration.options.common_qsw( durations_window_t, indices_elements_tx, indices_elements_rx, varargin{ : } );

            % iterate sound speed estimation options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).time_shift_ctr = time_shifts_ctr( index_object );
                objects( index_object ).factor_interp = factors_interp( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = sos_qsw( durations_window_t, time_shifts_ctr, indices_elements_tx, indices_element_lb, indices_element_ub, varargin )

	end % methods

end % classdef sos_qsw < calibration.options.common_qsw
