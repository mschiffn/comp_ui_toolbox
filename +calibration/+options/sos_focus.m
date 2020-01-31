%
% superclass for all sound speed estimation options (refocusing method)
%
% author: Martin F. Schiffner
% date: 2020-01-22
% modified: 2020-01-23
%
classdef sos_focus < calibration.options.common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        time_shift_ctr ( 1, 1 ) physical_values.time = physical_values.second( 19 / 40e6 )      % time shift to waveform center
        interval_f ( 1, 1 ) math.interval { mustBeNonempty } = math.interval( physical_values.hertz( 2e6 ), physical_values.hertz( 6e6 ) )	% frequency interval for Fourier domain focusing
        factor_interp ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 30	% interpolation factor
        N_iterations_max ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 20	% maximum number of re-focusing experiments
        rel_RMSE ( 1, 1 ) double { mustBePositive, mustBeNonempty } = 1e-3                      % maximum permissible relative RMSE

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sos_focus( durations_window_t, time_shifts_ctr, indices_element_lb, indices_element_ub, intervals_f, factors_interp, N_iterations_max, rel_RMSEs, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class physical_values.time for durations_window_t

            % property validation function ensures class physical_values.time for time_shifts_ctr

            % ensure positive integers for indices_element_lb
            mustBeInteger( indices_element_lb );
            mustBePositive( indices_element_lb );

            % ensure positive integers for indices_element_ub
            mustBeInteger( indices_element_ub );
            mustBePositive( indices_element_ub );

            % ensure strictly increasing bounds
            if any( indices_element_lb( : ) >= indices_element_ub( : ) )
                errorStruct.message = 'indices_element_ub must exceed indices_element_lb!';
                errorStruct.identifier = 'sos_focus:LowerBoundsExceedUpperBounds';
                error( errorStruct );
            end

            % property validation function ensures class math.interval for intervals_f

            % ensure nonempty factors_interp
            if nargin < 6 || isempty( factors_interp )
                factors_interp = repmat( 30, size( durations_window_t ) );
            end

            % property validation function ensures nonempty positive integers for factors_interp

            % ensure nonempty N_iterations_max
            if nargin < 7 || isempty( N_iterations_max )
                N_iterations_max = repmat( 20, size( durations_window_t ) );
            end

            % property validation function ensures nonempty positive integers for N_iterations_max

            % ensure nonempty rel_RMSEs
            if nargin < 8 || isempty( rel_RMSEs )
                rel_RMSEs = repmat( 1e-3, size( durations_window_t ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( durations_window_t, time_shifts_ctr, indices_element_lb, indices_element_ub, intervals_f, factors_interp, N_iterations_max, rel_RMSEs );

            %--------------------------------------------------------------
            % 2.) create sound speed estimation options
            %--------------------------------------------------------------
            % define indices of tx / rx elements (cross-correlation method requires contiguous indices, refocusing requires equal tx and rx indices)
            indices_elements_tx_rx = cell( size( durations_window_t ) );
            for index_object = 1:numel( durations_window_t )
                indices_elements_tx_rx{ index_object } = ( indices_element_lb( index_object ):indices_element_ub( index_object ) );
            end

            % constructor of superclass
            objects@calibration.options.common( durations_window_t, indices_elements_tx_rx, indices_elements_tx_rx, varargin{ : } );

            % iterate sound speed estimation options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).time_shift_ctr = time_shifts_ctr( index_object );
                objects( index_object ).interval_f = intervals_f( index_object );
                objects( index_object ).factor_interp = factors_interp( index_object );
                objects( index_object ).N_iterations_max = N_iterations_max( index_object );
                objects( index_object ).rel_RMSE = rel_RMSEs( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = sos_focus( durations_window_t, time_shifts_ctr, indices_element_lb, indices_element_ub, intervals_f, factors_interp, N_iterations_max, rel_RMSEs, varargin )

	end % methods

end % classdef sos_focus < calibration.options.common
