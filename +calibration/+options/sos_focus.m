%
% superclass for all sound speed estimation options (refocusing method)
%
% author: Martin F. Schiffner
% date: 2020-01-22
% modified: 2020-02-03
%
classdef sos_focus < calibration.options.common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        time_shift_ctr ( 1, 1 ) physical_values.time = physical_values.second( 19 / 40e6 )      % time shift to waveform center
        interval_f ( 1, 1 ) math.interval { mustBeNonempty } = math.interval( physical_values.hertz( 2e6 ), physical_values.hertz( 6e6 ) )      % frequency interval for Fourier domain focusing
        anti_aliasing ( 1, 1 ) scattering.options.anti_aliasing { mustBeNonempty } = scattering.options.anti_aliasing_boxcar                    % spatial anti-aliasing filter options
        relative_bandwidth_lb ( 1, 1 ) double { mustBeNonnegative, mustBeLessThanOrEqual( relative_bandwidth_lb, 1 ), mustBeNonempty } = 1      % lower bound on the relative bandwidth
        N_iterations_max ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 20	% maximum number of re-focusing experiments
        factor_interp ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 30	% interpolation factor
        rel_RMSE ( 1, 1 ) double { mustBePositive, mustBeNonempty } = 1e-3                      % maximum permissible relative RMSE

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sos_focus( durations_window_t, time_shifts_ctr, intervals_f, anti_aliasings, relative_bandwidth_lbs, factors_interp, N_iterations_max, rel_RMSEs, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class physical_values.time for durations_window_t
            % property validation function ensures class physical_values.time for time_shifts_ctr
            % property validation function ensures class math.interval for intervals_f

            % ensure nonempty anti_aliasings
            if nargin < 4 || isempty( anti_aliasings )
                anti_aliasings = scattering.options.anti_aliasing_boxcar( size( durations_window_t ) );
            end

            % property validation function ensures class scattering.options.anti_aliasing for anti_aliasings

            % ensure nonempty relative_bandwidth_lbs
            if nargin < 5 || isempty( relative_bandwidth_lbs )
                relative_bandwidth_lbs = repmat( 0.65, size( durations_window_t ) );
            end

            % property validation function ensures nonnegative values less than or equal to 1 for relative_bandwidth_lbs

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
            auxiliary.mustBeEqualSize( durations_window_t, time_shifts_ctr, intervals_f, anti_aliasings, relative_bandwidth_lbs, factors_interp, N_iterations_max, rel_RMSEs );

            %--------------------------------------------------------------
            % 2.) create sound speed estimation options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@calibration.options.common( durations_window_t, varargin{ : } );

            % iterate sound speed estimation options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).time_shift_ctr = time_shifts_ctr( index_object );
                objects( index_object ).interval_f = intervals_f( index_object );
                objects( index_object ).anti_aliasing = anti_aliasings( index_object );
                objects( index_object ).relative_bandwidth_lb = relative_bandwidth_lbs( index_object );
                objects( index_object ).N_iterations_max = N_iterations_max( index_object );
                objects( index_object ).factor_interp = factors_interp( index_object );
                objects( index_object ).rel_RMSE = rel_RMSEs( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = sos_focus( durations_window_t, time_shifts_ctr, intervals_f, factors_interp, N_iterations_max, rel_RMSEs, varargin )

	end % methods

end % classdef sos_focus < calibration.options.common
